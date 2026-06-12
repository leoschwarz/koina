# Re-exporting old Prosit models for CPU

The Prosit **2019/2020** intensity & iRT cores are TF1 Keras graphs whose GRUs are baked as the
GPU-only `CudnnRNN` op (no CPU kernel, no fallback subgraph), so they fail to load on a CPU host.
All Prosit/Prosit-XL/pfly models from **2023 onward already run on CPU unmodified** (they carry a
`standard_gru` CPU-fallback subgraph that Triton selects automatically) — only these 7 old cores
need re-exporting:

| Model | Zenodo record | Arch |
|---|---|---|
| **Prosit_2019_intensity** ✅ piloted | 7565518 | intensity |
| Prosit_2020_intensity_CID | 7681435 | intensity |
| Prosit_2020_intensity_HCD | 7681452 | intensity |
| Prosit_2020_intensity_TMT | 7681498 | intensity |
| Prosit_2019_irt | 7671314 | iRT |
| Prosit_2019_irt_supplement | 7671350 | iRT |
| Prosit_2020_irt_TMT | 7681553 | iRT |

## The recipe (proven on Prosit_2019_intensity → cosine 1.000000 vs GPU)

Conversion is **correct by construction**: rebuild the architecture in TF2 Keras with
`GRU(reset_after=True)` and load the original trained weights. The single non-obvious step is the
**CuDNNGRU → GRU weight-layout conversion** (keras [PR #9112](https://github.com/keras-team/keras/pull/9112)),
applied per gate block:

```python
kernel          = np.hstack([k.T.reshape(k.shape, order="F") for k in np.hsplit(kernel, 3)])   # input kernels
recurrent_kernel = np.hstack([k.T for k in np.hsplit(recurrent_kernel, 3)])                     # recurrent: transpose
bias             = bias.reshape(2, 3 * units)                                                   # keep both biases (reset_after)
```

Skipping the recurrent-kernel transpose silently yields ~0.54 cosine (looks plausible, is wrong) —
**always gate on numerical parity, never on "it loads".** `tf2onnx` of the frozen graph is a
confirmed dead end (it mis-converts the cuDNN GRU and is numerically wrong).

The re-exported SavedModel uses standard GRU ops → runs on CPU, and on GPU uses cuDNN via the
implementation_selector dual-path (same as the 2023+ models). Its serving signature is patched to
match the original (`peptides_in:0`, `collision_energy_in:0`, `precursor_charge_in:0` →
`out/Reshape:0`) so it **drops into koina with no config or ensemble change** (stays on the
`tensorflow_savedmodel` backend; the existing `KIND_AUTO` placement works).

## How to run

```bash
docker build -t koina-convert -f tools/cpu_reexport/Dockerfile tools/cpu_reexport

# 1. obtain the original SavedModel (downloaded by koina at startup, or unzip the Zenodo .savedmodel.zip)
# 2. re-export:
docker run --rm -v "$PWD:/w" koina-convert \
  python /w/tools/cpu_reexport/reexport_prosit_intensity.py \
    /w/models/Prosit/Prosit_2019_intensity_core/1/model.savedmodel /w/out/cpu_savedmodel

# 3. drop /w/out/cpu_savedmodel in as the core's model.savedmodel, run koina with KOINA_FORCE_CPU=1,
#    and validate end-to-end vs the public GPU server (cosine must be ~1.0).
```

## Status / scope

- `reexport_prosit_intensity.py` is the **intensity** architecture (embedding → BiGRU → GRU →
  attention → meta-fusion → decoder GRU → decoder-attention → TimeDistributed dense). It should
  cover all 4 intensity cores; the 2020 CID/HCD/TMT variants must each be parity-checked (TMT/CID
  differ only in training data / alphabet, not topology — verify).
- The **iRT** cores use the simpler `PrositRetentionTimePredictor` topology (no decoder/meta
  branch: embedding → BiGRU → GRU → attention → Dense(relu) → Dense(1)). Same weight-conversion
  recipe; a sibling `reexport_prosit_irt.py` is the obvious next step (not yet written).

## ⚠️ Hosting handoff (required, maintainer-only)
A re-exported model can't be shipped by editing this repo — koina downloads weights from Zenodo at
startup and model files are gitignored. Publishing requires **uploading the new SavedModel to a new
Zenodo record and bumping the core's `1/.zenodo` URL + MD5**. That is a maintainer action.
