# Original agent and maintenance boundaries

The agent pipeline is `timeclaw/pipeline/runner.py`. It prepares the visible
series and optional text/images, builds task context, and invokes the single
Executor's ReAct controller. ReAct means repeated LLM decision → tool execution
→ observation, ending in a task-valid final answer. This is distinct from a
direct single-model forecast call.

Learning enters through `timeclaw.benchmark.run_learning --train` and
`train_loop.run_train_loop`. Training targets are available to the training
evaluator only. The native executor explores real candidate branches and records
the compared complete paths. Tool dropout varies tool availability to encourage
alternative paths; it does not synthesize results for unexecuted branches.

`memory/hierarchical.py` is included in full. A Note records one training
experience. MEMORY is the rule store distilled from multiple Notes. Derived
TOOLS/SKILLS/SKILLS_DECISION guidance is updated from that store. Ten new Notes
per scope trigger the batch stage, thirty governs history snapshots, and final
draining handles pending work. These are distinct counters, not extra per-sample
learning stages. After multi-branch exploration reaches its budget, a sole valid,
actually scored path can record a `single_execution` Note: actual tools and scores,
no winner/comparison. These Notes count toward the normal scoped batch trigger,
but cannot support relative tool-selection/failure rules or winner frequencies.
Unscored or zero-candidate attempts remain failed evidence. No new model weights are trained by this lifecycle.

Inference uses the frozen memory and `PipelineRunner.run_inference`, then
`run(..., mode="test")`. Test scores do not select online tools, and inference
does not create learning Notes or distill on shutdown. The evaluation wrapper
uses targets only after the prediction is returned. Legacy sample APIs may use
the output array length as the public horizon when no explicit horizon exists;
do not confuse known horizon metadata with access to future values.

`core/accounting.py` records logical LM calls and actual transport attempts,
including nested tool calls through MCP. Unknown failed-call usage is not zero
cost. This is maintenance instrumentation, not a memory compiler. The generic
fatal evidence error lives in `core/errors.py`; neither module imports a research
runtime. Original handwritten task rules remain handwritten, not learned memory.

Artifact roots are selected once per CLI process using the configured workspace.
An explicit learning CLI workspace overrides the environment, which overrides
config; an omitted flag never substitutes the CLI's default for the config.
`core.trace_logger.get_trace_path` owns trace creation, diagnostic/evidence reads
and Note references, so readers cannot retain a hardcoded default trace root.
The historical implementation binds module-level roots; do not concurrently use
multiple workspaces inside one Python process. Use separate processes instead.

The paper benchmark task owner is `benchmark/task_registry.py`: default
`paper30_original` has 17 tasks / 30 metric cells, including short weather
indicators and MCQA. The earlier `timeclaw18` inventory is an explicit extension. Pipeline tools remain reusable building blocks, not extra tasks.
`benchmark/source_admission.py` owns declared source support, data isolation and
sealed training lineage. The train loop admits sources before workspace changes,
and standalone distillation reseals its updates. Benchmark and CSV inference
verify existing seals and disable learning; benchmark inference also checks the
frozen evaluation inputs. Unknown artifacts are rejected, never relabeled.

The public source is scoped to main experiments. Legacy toy benchmark,
post-hoc ReasonerAgent, fusion/tokenization/scheduler scaffolds, data builders,
case studies and ablation entrypoints are excluded. Internal constructor controls
used to isolate unit tests are not alternative experimental runners. The main
LLM baseline uses the same task/scoring owners; it is a main-table comparison,
not a new agent or an optional-memory experiment.


Public job entrypoints hold `source_admission.workspace_access` throughout
execution and cleanup. Learning/distillation requires exclusive write access;
frozen inference permits shared readers. Receipt validation remains necessary:
a file lock does not certify source ancestry, nor prevent arbitrary external
programs from editing files. Busy jobs fail explicitly before backend startup.
