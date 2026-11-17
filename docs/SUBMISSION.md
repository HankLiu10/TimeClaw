# Source submission guide

## What is delivered

This is the repaired TimeClaw1 main-experiment source: default 17 paper tasks / 30 metrics (explicit 18-task extension),
native ReAct exploration, tool dropout, complete hierarchical memory,
frozen inference, main baseline, shared scoring and their regression tests.
No dataset builders, ablations, case studies, synthetic demonstration generators
or TimeClaw2 runtime are included. Raw data, old training checkpoints, Notes,
learned memory, prediction records, weights, private configs and old Git history
are not submission contents. Unit-test fixtures are software checks, not old
training products or paper results.

## Starting from source

Follow README.md for Linux/Python 3.10 installation, the locked environment,
pinned MTBench setup and evaluation-data download. Downloaded inputs and optional
weights remain outside version control. Supply an independently collected,
admissible corpus and use a new workspace for learning; official evaluation
samples are never substituted for missing training data. Credentials come from
the process environment, not source files. The default profile is a tested
software profile, not the exact historical paper environment.

`setup_mtbench.py` installs into a temporary sibling directory and only exposes
the final destination after checkout succeeds. A failed fetch can be retried
without a partial installation in the final path. An existing dirty or wrong-
revision checkout is rejected without overwriting user files.

## Verification commands

From the unpacked source root:

```bash
sha256sum -c MANIFEST.sha256
python -m pip check
python -m ruff check timeclaw scripts tests --select E9,F821,F822,F823
python -m pytest -q --timeout=20
```

The manifest lists delivered source files, excluding its own self-hash and the
build system's generated PKG-INFO. It is not a runtime-data manifest. Regression
checks block external network connections except loopback; tests need neither
paid credentials nor neural weights. Pin metadata and upstream setup were also
checked separately against the public sources on 2026-11-09.

## Runtime outcomes and model capabilities

The CSV command preserves `result.json` and exits with an error if native quality
admission fails, a prediction is absent, or the result contains an execution
error. A valid zero forecast remains successful. Cleanup still runs. Benchmark
batches retain their own failed rows and exit unsuccessful when appropriate.

The historical TimerAdapter wrapper actually targets Sundial Base 128M and
advertises forecasting only. Unimplemented operations do not load weights or
produce synthetic outputs. See MODELS.md for optional model environments.

## Official publication checklist

- Use the clean, tested source snapshot and verify the Git commit contains no
  private credentials, datasets, weights, memory or old experiment products.
- Inspect all remote branches/history, including managed PR references. Normal
  branch ancestry has now been cleaned with the original README preserved,
  but PR #1 and credential disposition remain open. Remote cleanup alone
  cannot establish that exposed credentials are inactive.
- Use `paper30_original` for the paper task/metric inventory. The `timeclaw18`
  extension is separately named; task alignment is not historical score reproduction.
- Preserve the declared observation-time attachment and exact original input
  checksums. All eight pools now pass source support without guessed times or
  exclusions; see DATA_READINESS.md.
- Publish an admissible independent learning corpus separately if full paper
  reproducibility is intended; the current source has no verified download URL.
- Revalidate the package and install checks after any release change, and verify
  the public GitHub view after the maintainer authorizes and performs upload.

## Acceptance boundary

Current software evidence is in RELEASE_STATUS.md and VALIDATION.json. Software
submission acceptance is separate from historical numerical-paper reproduction;
old products are not required for this delivery. Every optional GPU model,
platform and full benchmark score is not certified by the core suite.
Submit a clean source snapshot, without copying a research workspace or an old
.git directory. No automatic push or rewrite of remote history is part of these
checks.
