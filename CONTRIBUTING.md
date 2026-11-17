# Contributing to TimeClaw

Use GitHub Issues for reproducible bugs and pull requests for changes. Include
the commit/version, Python and optional-model environment, exact command,
expected behavior and a minimal sanitized error trace. Never post API keys,
private data or full private prompts in an issue.

Install the core/development environment and pinned MTBench dependency as in
README.md. Before submitting a change, run:

```bash
python -m pip check
python -m ruff check timeclaw scripts tests --select E9,F821,F822,F823
LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q --timeout=20
```

Add a regression at the actual failing interface when repairing a bug. Keep
external model calls and large weight downloads out of core tests. Document
optional dependencies and model-specific checks separately.

The task roster is owned by `timeclaw/benchmark/task_registry.py`. Train and
inference share task preparation and scoring owners. Changes must preserve
independent-source admission, supervision-free test requests, sealed memory,
workspace access rules and recorded failures. Do not add a second hidden roster,
select better test runs per sample or encode sample IDs/paths as prediction
rules. Declare public mathematical task rules separately from learned memory.

Use a new workspace when changing input ancestry. Keep datasets, learned files,
model weights, local configs and experimental records outside the source tree.
Do not add dataset builders, ablations, case studies or unrelated research
runners to this main-experiment distribution without an explicit scope change.

Maintain README commands and relevant docs when changing public interfaces.
For a release, regenerate MANIFEST.sha256, build source/wheel artifacts and
verify the unpacked source; a passing checkout alone is not package acceptance.
