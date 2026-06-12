# Re-exporting old Prosit models for CPU

The Prosit **2019/2020** intensity & iRT cores are TF1 Keras graphs whose GRUs are baked as the
GPU-only `CudnnRNN` op (no CPU kernel, no fallback subgraph), so they fail to load on a CPU host.
All Prosit/Prosit-XL/pfly models from **2023 onward already run on CPU unmodified** (they carry a
`standard_gru` CPU-fallback subgraph that Triton selects automatically) — only these 7 old cores
need re-exporting. **All 7 are re-exported and validated on CPU — numerically identical to the
GPU server (cosine 1.0 / iRT Δ < 1e-4).**

| Model | Zenodo | Arch | re-export args (after SRC OUT) | CPU vs GPU |
|---|---|---|---|---|
| Prosit_2019_intensity | 7565518 | intensity | *(defaults)* | cosine 1.000000 |
| Prosit_2020_intensity_CID | 7681435 | intensity | *(defaults)* | cosine 1.000000 |
| Prosit_2020_intensity_HCD | 7681452 | intensity | *(defaults)* | cosine 1.000000 |
| Prosit_2020_intensity_TMT | 7681498 | intensity | `23 frag` | cosine 1.000000 |
| Prosit_2019_irt | 7671314 | iRT | `22 32 sequence_integer` | Δ 6e-6 |
| Prosit_2019_irt_supplement | 7671350 | iRT | `22 32 sequence_integer` | Δ 1.5e-5 |
| Prosit_2020_irt_TMT | 7681553 | iRT | `23 16 peptides_in:0` | Δ 2.3e-5 |

Intensity defaults = vocab 22 + 3 inputs; `23 frag` = vocab 23 + the extra `fragmentation_type_in:0`
input (2020 TMT). The decoder-attention dense var (`dense_1`/`dense_19`/…) is auto-detected.
Validate TMT models with TMT-labeled peptides (`[UNIMOD:737]`).

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

All 7 done and CPU-validated. Two scripts cover them:
- `reexport_prosit_intensity.py` — intensity architecture (embedding → BiGRU → GRU → attention →
  meta-fusion → decoder GRU → decoder-attention → TimeDistributed dense). Covers all 4 intensity
  cores; `23 frag` handles the 2020 TMT variant (vocab 23 + `fragmentation_type_in:0`, meta order
  = collision_energy, precursor_charge, fragmentation_type per dlomix `META_DATA_KEYS`).
- `reexport_prosit_irt.py` — `PrositRetentionTimePredictor` topology (embedding → BiGRU → GRU →
  attention → Dense(relu) [pep_dense1] → Dense(1) [prediction]); parameterized by vocab, embedding
  dim, and the ensemble's input name.

Each rebuilt SavedModel was dropped into its `models/Prosit/<name>_core/1/` and validated
end-to-end against the public GPU server (see table above). To reproduce: re-export, place the
output as the core's `model.savedmodel` plus the original `.savedmodel.zip` as `zenodo.zip` (so
koina skips re-download), run with `KOINA_FORCE_CPU=1`, and parity-check vs the public server.

## ⚠️ Hosting handoff (required, maintainer-only)
A re-exported model can't be shipped by editing this repo — koina downloads weights from Zenodo at
startup and model files are gitignored. Publishing requires **uploading the new SavedModel to a new
Zenodo record and bumping the core's `1/.zenodo` URL + MD5**. That is a maintainer action.

**Current branch state (pre-PR testing).** The 7 cores' `1/.zenodo` files now point at a temporary
FGCZ test host (`https://fgcz-ms.uzh.ch/public/koina/20260612_cpu_build/`) serving the re-exported
CPU SavedModels, so the branch is testable end-to-end on CPU as-is. For the upstream PR a maintainer
re-hosts the same zips on Zenodo and swaps each `.zenodo` URL — **the md5 stays identical** (same
zip), so it's a one-line-per-model change.

**Licensing (re-hosting is permitted).** The source Zenodo records are all open and permit
derivative works + redistribution: `Prosit_2019_intensity` (record 7565518) is **Apache-2.0**; the
other six are **CC-BY-4.0**. They were published by the koina maintainer (L. Lautenbacher, TUM) —
same institution as the upstream `kusterlab/prosit` (Apache-2.0). Conditions when re-hosting: keep
**the same license per record** (Apache for 2019_intensity, CC-BY-4.0 for the rest), attribute
Gessulat et al. 2019 + link the source DOI, and **mark the record as modified** ("re-exported with
a CPU-compatible GRU"). No upstream `NOTICE` file exists, so nothing to propagate. (Not legal
advice; the publish decision is the maintainers'.)
