# Optional forecasting models

The core installation supports the agent and statistical tools. Neural-model
packages and weights are separate. Discovery lists registered tools; it does
not certify their environments or checkpoints. Respect upstream model terms.

A prior engineering validation used the native past-trend task on CPU without managed neural services. The complete original catalog remains
available; selecting an optional tool still requires its dependencies.

For Chronos, install the extra in the intended environment:

```bash
python -m pip install -e '.[sbm-chronos]'
```

Then configure a local file, for example:

```yaml
sbm:
  device: cuda:0
  managed_server_enabled: true
  managed_server_models: [chronos2]
  managed_server_replicas: 1
```

Other native adapters, including Moirai, TimesFM, Sundial and Aurora,
remain in `timeclaw/sbm/`. Extras are declared in `pyproject.toml`. The TimesFM
adapter uses the original TimesFM 2.0 API and its extra pins `timesfm==1.3.0`;
it does not install the unrelated moving Transformers repository. Inspect the
adapter for its checkpoint ID and additional setup. The core dependency lock
does not freeze these separate environments. Conflicting models should use
separate environments via `sbm.managed_server_python`, keyed by registered model
name. No automatic model substitution is added when an environment is missing.

The historical `timer_adapter.py` / `TimerAdapter` names wrap
`thuml/sundial-base-128m`, registered as `sundial_base_128m`; they do not certify a
separate Timer model. Its declared capability is forecasting. Imputation and
anomaly detection are not implemented by that adapter; unsupported imputation
raises before loading weights. This is a capability correction, not a replacement
prediction model.

Managed services start once per learning job and are shared across samples.
`managed_server_enabled: false` explicitly leaves service management disabled.
For standalone service configuration, see
`python -m timeclaw.sbm.server --help`. Report setup/loading separately from warm
request processing when measuring latency. This release validation did not run
GPU model loading or certify every adapter's upstream checkpoint compatibility.
