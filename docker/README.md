# Container images

The harness expects two images, pulled through the DaoCloud Docker Hub mirror:

```
docker.m.daocloud.io/vllm/vllm-openai:latest
docker.m.daocloud.io/lmsysorg/sglang:latest
```

Use `scripts/pull_images.sh`; it checks disk, retries, and records image IDs and
`RepoDigests` in `images.lock.json`.

## Pin for reproducibility

`latest` is convenient but moves. For a run you intend to compare later:

```bash
VLLM_TAG=v0.10.1 SGLANG_TAG=v0.4.10 ./scripts/pull_images.sh
```

Then set the same tags in `configs/lab.json` (or via `VLLM_IMAGE` /
`SGLANG_IMAGE`) so the harness uses exactly those images.

## Manual fallback

```bash
docker pull docker.m.daocloud.io/vllm/vllm-openai:latest
docker tag  docker.m.daocloud.io/vllm/vllm-openai:latest vllm/vllm-openai:latest
```

If the mirror is down, `ALLOW_DOCKERHUB_FALLBACK=1 ./scripts/pull_images.sh`
tries Docker Hub and retags to the mirror name. Record which path was used; the
digest is what matters.

## Offline transfer

If the target host cannot reach a registry:

```bash
# on a machine with access
docker pull docker.m.daocloud.io/vllm/vllm-openai:latest
docker save docker.m.daocloud.io/vllm/vllm-openai:latest | gzip > vllm.tar.gz

# on the target
gunzip -c vllm.tar.gz | docker load
```

`docker save` preserves the image ID; the tag is included in the archive.

## Model cache

Images mount the named volume `inference-exp-hf` at
`/root/.cache/huggingface`. To pre-fetch the model instead of waiting inside a
timed engine startup:

```bash
./scripts/fetch_model.sh
```

The script uses the vLLM image's `huggingface_hub` and honors `MODEL`,
`MODEL_IMAGE`, `HF_ENDPOINT`, and `HF_TOKEN`.
