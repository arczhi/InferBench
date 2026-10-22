PY ?= python3
ENGINE ?= all
PROFILE ?= quick

.PHONY: help test survey pull fetch-model fetch-model-ms dry-run smoke-kv smoke-cliff kv-quick cliff-quick kv-full cliff-full clean

help:
	@echo "make survey        # read-only host survey (run this first)"
	@echo "make pull          # pull vLLM + SGLang through docker.m.daocloud.io"
	@echo "make fetch-model   # pre-download the model into the shared volume"
	@echo "make fetch-model-ms # download the model from ModelScope to /data/models"
	@echo "make test          # stdlib unit tests (safe on any machine)"
	@echo "make dry-run       # print docker commands and level plans"
	@echo "make smoke-kv      # 1 arm, 8K, N=1,2"
	@echo "make smoke-cliff   # baseline arms, 8K, N=1,2,4"
	@echo "make kv-quick      # ENGINE=all PROFILE=quick"
	@echo "make cliff-quick   # ENGINE=all PROFILE=quick"
	@echo "make kv-full       # long; run inside tmux"
	@echo "make cliff-full    # long; run inside tmux"

test:
	$(PY) -m unittest discover -s tests -v

survey:
	$(PY) scripts/survey_l4.py --json-out survey_$(shell hostname).json

pull:
	./scripts/pull_images.sh

fetch-model:
	./scripts/fetch_model.sh

fetch-model-ms:
	./scripts/fetch_model_modelscope.sh

dry-run:
	$(PY) -m exp_kv.run --profile smoke --engine vllm --dry-run
	$(PY) -m exp_cliff.run --profile smoke --engine vllm --dry-run

smoke-kv:
	$(PY) -m exp_kv.run --profile smoke --engine vllm

smoke-cliff:
	$(PY) -m exp_cliff.run --profile smoke --engine vllm

kv-quick:
	$(PY) -m exp_kv.run --profile $(PROFILE) --engine $(ENGINE)

cliff-quick:
	$(PY) -m exp_cliff.run --profile $(PROFILE) --engine $(ENGINE)

kv-full:
	$(PY) -m exp_kv.run --profile full --engine $(ENGINE)

cliff-full:
	$(PY) -m exp_cliff.run --profile full --engine $(ENGINE)

clean:
	./scripts/cleanup.sh --runs-keep 10
