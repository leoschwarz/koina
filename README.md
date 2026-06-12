# Koina

## Accessing a public server
### cURL
Here is an example HTTP request using only cURL sending a POST request to with a JSON body. You can find examples for all available models at https://koina.wilhelmlab.org/. 

```bash
curl "https://koina.wilhelmlab.org/v2/models/Prosit_2019_intensity/infer" \
 --data-raw '
{
  "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
  "inputs": [
    {"name": "peptide_sequences",   "shape": [2,1], "datatype": "BYTES", "data": ["LGGNEQVTR","GAGSSEPVTGLDAK"]},
    {"name": "collision_energies",  "shape": [2,1], "datatype": "FP32",  "data": [25,25]},
    {"name": "precursor_charges",    "shape": [2,1], "datatype": "INT32", "data": [1,2]}
  ]
}'
```

The output of an HTTP request is always a JSON object. The `outputs` key contains the outputs the model provides. In this case, there are three outputs: `annotation,` `mz`, and `intensities`. For other models, the keys change.

```json
{
    "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
    "model_name": "Prosit_2019_intensity",
    "model_version": "1",
    "parameters": {
        "sequence_id": 0,
        "sequence_start": false,
        "sequence_end": false
    },
    "outputs": [
        {
            "name": "annotation",
            "datatype": "BYTES",
            "shape": [
                2,
                174
            ],
            "data": [
                "y1+1",
                "y1+2",
                "y1+3",
                "b1+1",
                ...
                "y29+3",
                "b29+1",
                "b29+2",
                "b29+3"
            ]
        },
        {
            "name": "mz",
            "datatype": "FP32",
            "shape": [
                2,
                174
            ],
            "data": [
                175.11895751953125,
                -1.0,
                -1.0,
                114.09133911132812,
                ...
                -1.0,
                -1.0,
                -1.0,
                -1.0
            ]
        },
        {
            "name": "intensities",
            "datatype": "FP32",
            "shape": [
                2,
                174
            ],
            "data": [
                0.2463880330324173,
                -1.0,
                -1.0,
                0.006869315169751644,
                ...
                -1.0,
                -1.0,
                -1.0,
                -1.0
            ]
        }
    ]
}
```


### Python
For examples of how to access models using Python, you can check out [our OpenAPI documentation ](https://koina.wilhelmlab.org/docs/#overview).

## Hosting your own server

### Dependencies
Koina depends on [docker](https://docs.docker.com/engine/install/).
For GPU-accelerated inference it additionally requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).
CPU-only mode (no GPU required) is also supported — see below.

You can find an ansible script that installs all dependencies [here](docs/server/).

### How to run it
After installing the dependencies, you can pull the docker image and run it. The time it takes to pull the image depends on your connection speed. The first time, it might take up to 5 min. Due to the layered design of Docker images, updating to the latest version will likely (depending on the amount of changes) only take seconds. When the server is first started, model files are downloaded from Zenodo. The duration of this also depends on connection speed but might take ~10 min as well. Once models are downloaded, the server startup takes ~2 minutes.

When using this docker image, you need to accept the terms in the [NVIDIA Deep Learning Container License](NVIDIA_Deep_Learning_Container_License.pdf)

#### With a GPU (recommended for production)

If you have multiple GPUs installed on your server, you can choose which one is used by modifying `--gpus '"device=0"'`.

```bash
docker run \
    --gpus '"device=0"' \
    --shm-size 8G \
    --name koina \
    -p 8500-8502:8500-8502 \
    -d \
    --restart unless-stopped \
    ghcr.io/wilhelm-lab/koina:latest
```

#### Without a GPU (CPU-only mode)

The same image runs on CPU — simply omit `--gpus`. The server detects the absence of a
GPU automatically (you can also force it with `-e KOINA_FORCE_CPU=1`). CPU inference is
slower, and to reduce RAM usage and startup time you should set `MODEL_PATTERN` to load
only the models you need rather than the default `MODEL_PATTERN=*` (which loads everything).

```bash
docker run \
    --shm-size 8G \
    --name koina \
    -e MODEL_PATTERN=Chronologer_RT \
    -p 8500-8502:8500-8502 \
    -d \
    --restart unless-stopped \
    ghcr.io/wilhelm-lab/koina:latest
```

**Most models run on CPU.** Confirmed working on CPU: the PyTorch models (e.g. Chronologer,
AlphaPept), the ms2pip (XGBoost) models, DeepLC / IM2Deep, all pre/post-processing steps,
and the **Prosit / Prosit-XL / pfly models from 2023 onward** (e.g. `Prosit_2023_intensity_timsTOF`,
the 2024/2025 intensity & iRT PTM variants, `Prosit_2023_intensity_XL_*`). These newer
TensorFlow models carry a CPU GRU fallback subgraph that Triton selects automatically on CPU
(verified numerically identical to GPU output).

The only models that currently **require a GPU** are the **older Prosit 2019 / 2020 models**
(`Prosit_2019_intensity`, `Prosit_2019_irt`, `Prosit_2020_intensity_{CID,HCD,TMT}`,
`Prosit_2020_irt_TMT`). Their TensorFlow graphs use the cuDNN-only `CudnnRNN` op (no CPU
kernel and no fallback branch), so they fail to load on a CPU host. The server stays healthy
and keeps serving every other model (it does not crash or restart-loop). Running these few on
CPU would require re-exporting them upstream with a CPU-compatible GRU.

If you want to stay up to date with the latest version of Koina we suggest you also deploy containrrr/watchtower.


```bash
docker run \
  -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --restart unless-stopped \
  containrrr/watchtower -i 30 --rolling-restart
```

## Adding your own model
You can find a detailed description under [docs/](docs/README.md). If you run into any issues please open an issue on GitHub.