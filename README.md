# Multi-agent MLGym

A hierarchical multi-agent layer built on top of Meta's MLGym. A supervisor agent plans a research workflow, creates specialised worker agents, runs them in a shared container, and aggregates their results. The repository keeps the full MLGym single-agent framework intact and adds the supervisor, the worker roles, a human-in-the-loop path, and the run scripts that drive them.

## Contents

* [What this is](#what-this-is)
* [Relationship to upstream MLGym](#relationship-to-upstream-mlgym)
* [The multi-agent architecture](#the-multi-agent-architecture)
* [How orchestration works](#how-orchestration-works)
* [Self-correction and critique](#self-correction-and-critique)
* [Worker roles and the orchestration pipeline](#worker-roles-and-the-orchestration-pipeline)
* [Tasks and benchmark](#tasks-and-benchmark)
* [Results](#results)
* [Repository structure](#repository-structure)
* [Getting started](#getting-started)
* [Running the multi-agent workflows](#running-the-multi-agent-workflows)
* [Single-agent MLGym (upstream)](#single-agent-mlgym-upstream)
* [Trajectory visualizer](#trajectory-visualizer)
* [Status and caveats](#status-and-caveats)
* [Credit and citation](#credit-and-citation)
* [License](#license)

## What this is

MLGym is a Gymnasium environment for machine learning research tasks. An agent works inside a container, reads a task, writes and runs code, validates its solution, and submits an artefact that is scored against a baseline. The upstream framework is single-agent: one model drives the whole task from start to finish.

This fork adds a layer above that agent. Instead of one agent solving a task end to end, a supervisor agent:

1. Plans a workflow: a strategy, a sequence of agents to run, and success criteria.
2. Creates worker agents one at a time, each with a role and instructions.
3. Runs each worker inside the same container so later workers can see and build on earlier work.
4. Tracks per-agent steps, exit status, cost, and tokens, then aggregates them into a single results file.

The supervisor, the worker agents, and the human-in-the-loop variant all reuse MLGym's existing agent loop, tool system, container management, and trajectory format. The new code is the planning and routing environment, the supervisor agent classes, the worker prompt templates, and the run scripts.

## Relationship to upstream MLGym

This is a fork of [facebookresearch/MLGym](https://github.com/facebookresearch/MLGym). The base framework, the task suite, the data, the tools, the backend, the trajectory visualizer, and almost all of the `mlgym/` package come from the upstream project and carry Meta copyright headers. The MLGym paper is Nathani et al., 2025 (arXiv:2502.14499).

The commit history makes the lineage explicit. The upstream history runs up to the 0.1.1 ruff and mypy migration. On top of that, a single commit, "Add supervisor agent implementations and orchestration features," introduces everything described in this README. If you want to see exactly what was added on top of MLGym, that commit is the boundary.

The multi-agent work added here is:

* `mlgym/agent/supervisor_agent.py`, `mlgym/agent/research_supervisor.py`, `mlgym/agent/decoupled.py`, `mlgym/agent/supervisor_aware.py`
* `mlgym/environment/supervisor_env.py`, `mlgym/environment/supervisor_env_mlgym.py`
* `mlgym/constants.py` (supervisor and worker step limits)
* The supervisor, worker, and orchestration configs under `configs/agents/`
* The new top-level run scripts (`run_supervisor_mlgym.py`, `run_supervisor_basic.py`, `run_HIL.py`, `multi_run.py`, and the experiment drivers)

## The multi-agent architecture

There are four agent classes added on top of MLGym's `BaseAgent`, plus two supervisor environments.

### Agent classes

| Class | File | Role |
| --- | --- | --- |
| `SupervisorAgent` | `mlgym/agent/supervisor_agent.py` | Thin `BaseAgent` subclass. Drives the supervisor loop by emitting `plan_workflow`, `create_agent`, and `complete_workflow` commands. The planning logic lives in its prompt and in the supervisor environment. |
| `ResearchSupervisor` | `mlgym/agent/research_supervisor.py` | A heavier supervisor that analyses the task, scores worker code quality, detects the current research phase, decides when intervention is needed, and assesses submission readiness. Driven by the `research_supervisor.yaml` config. |
| `DecoupledAgent` | `mlgym/agent/decoupled.py` | A worker that proposes an action but does not execute it. Execution is left to the run loop or environment, which is what enables a propose-then-verify pattern. |
| `SupervisorAwareAgent` | `mlgym/agent/supervisor_aware.py` | A worker that can pause and talk to a supervisor using the `ask_supervisor` and `submit_request` commands, used by the human-in-the-loop script. |

### Supervisor environments

Both environments subclass `MLGymEnv` and add a planning and routing layer. They share the same three supervisor commands but differ in maturity.

* `SupervisorEnvMLGym` (`supervisor_env_mlgym.py`) is the one the main supervisor script uses. It validates supervisor arguments, registers supervisor commands through MLGym's tool system, pre-allocates agent tracking to avoid race conditions, loads the enhanced worker template per agent, builds a collaborative context describing previous agents, escapes worker instructions into container environment variables, runs each worker, captures per-agent cost and token stats, and aggregates everything into a `results.json` with a `supervisor_stats` block.
* `SupervisorEnv` (`supervisor_env.py`) is a simpler earlier version that uses `DecoupledAgent` workers and a fixed worker action list. It covers the same plan, create, run, complete cycle with less tracking and no per-agent cost aggregation.

### Supervisor commands

The supervisor only ever issues one of three commands per step, in this order:

```
plan_workflow [strategy: "..."] [agents: "agent1,agent2,agent3"] [criteria: "..."]
create_agent <name> '<instructions>'
complete_workflow
```

`get_available_actions` gates these by state: only `plan_workflow` is allowed until a workflow exists, then `create_agent` until all planned agents have run, then `complete_workflow`.

## How orchestration works

The main loop lives in `run_supervisor_mlgym.py`. The supervisor and the workers run in the same container, sequentially.

```
                       run_supervisor_mlgym.py
                                 |
                                 v
                  +-----------------------------+
                  |     SupervisorAgent (LLM)    |
                  |  plan -> create -> complete  |
                  +-----------------------------+
                                 |
              proposes one supervisor command per step
                                 v
                  +-----------------------------+
                  |     SupervisorEnvMLGym       |
                  |   (subclass of MLGymEnv)     |
                  +-----------------------------+
                        |            |        \
            plan_workflow  create_agent     complete_workflow
                        |            |              |
            store plan   instantiate worker     aggregate results
            + agent list  in shared container    + save results.json
                                     |
                                     v
                  +-----------------------------+
                  |   Worker agent (BaseAgent)   |
                  |  edit / run / validate /     |
                  |  submit, up to N steps       |
                  +-----------------------------+
                                     |
                 results, cost, tokens, exit status
                                     |
                                     v
                  collaborative context passed to the
                  next worker (previous outputs, position,
                  next role, objective, success criteria)
```

Step by step:

1. The supervisor receives the task description and the current workflow state, then proposes a command.
2. On `plan_workflow`, the environment parses a strategy, an agent sequence, and success criteria out of the command text and stores an `AgentPlan` per agent.
3. On `create_agent`, the environment builds a fresh worker config from the enhanced worker template, injects the role and instructions and a collaborative context (a summary of the last few completed agents, the worker's position in the sequence, the next agent's role, the workflow objective, and the success criteria) into the container as environment variables, then runs the worker until it submits or hits its per-agent step limit. The worker runs in the same container, so files written by earlier workers are visible.
4. Per-agent cost, tokens sent and received, API calls, steps, and exit status are recorded. Each worker also writes its own trajectory under `trajectories/<user>/<run_name>/<role_n>/`.
5. On `complete_workflow`, the environment writes a combined `results.json` that lists every agent's scores plus a `supervisor_stats` block (total agents, total steps, total agent cost, supervisor cost, combined cost and tokens, average score) and the task baseline.

Step budgets are separate for the supervisor and the workers, set in `mlgym/constants.py`: `DEFAULT_MAX_SUPERVISOR_STEPS = 10`, `DEFAULT_MAX_AGENTS_PER_WORKFLOW = 10`, `DEFAULT_MAX_STEPS_PER_AGENT = 30`. They are overridable from the command line.

## Self-correction and critique

The "self-reflection and cross-agent critique" in this system comes from a few concrete mechanisms rather than one module.

* Format self-correction. Inherited from MLGym's `BaseAgent`: when a model produces output that does not parse into a discussion plus a single command, `forward_with_error_check` and the requery path ask the model to fix its own output before the step counts. This applies to the supervisor and every worker.
* Supervisor critique and intervention. `ResearchSupervisor` scores worker code on syntax, completeness, correctness, and efficiency, tracks which research phase the worker is in (understanding, implementation, testing, optimization, submission), and fires intervention triggers when progress stalls, when there is no implementation late in the run, when the worker is stuck exploring, or when edit attempts repeat. It then emits targeted guidance. The thresholds and patterns are configurable from YAML.
* Cross-agent handoff. Each worker is told what previous workers produced and is instructed to read their outputs, build on what worked, avoid repeating failures, and leave documentation for the next worker. This is the worker template's collaboration contract, reinforced by the `previous_agent_context` the environment injects.
* Human-in-the-loop approval. With `SupervisorAwareAgent` and `run_HIL.py`, a worker can call `ask_supervisor` to ask for guidance or `submit_request` to ask for approval before submitting. A human at the terminal approves, rejects with feedback, or answers, and the feedback flows back into the worker's next observation.

## Worker roles and the orchestration pipeline

Worker behaviour is set entirely by prompt templates under `configs/agents/`. The supervisor can name arbitrary roles at plan time, but the repository also ships fixed role configs for a feature-engineering, modeling, validation pipeline in `configs/agents/orchestration/`:

* `feature_eng.yaml`: cleans and transforms the dataset, writes engineered files and a `feature_insights.txt` for the next agent.
* `modeling.yaml`: looks for the engineered files and insights, trains and compares models, logs experiments to `modeling_log.txt`, and saves predictions.
* `validation.yaml`: checks every prior output, runs `validate`, records its decision in `validation_log.txt`, and makes the final `submit`.

These agents coordinate through files in the shared container (engineered datasets, insight files, logs) rather than through any message bus. The worker templates also include guardrails learned from runs, for example not opening large CSVs directly, always validating before submitting, and submitting a working solution when few steps remain.

Other worker and supervisor configs:

* `worker_template.yaml`, `worker_template_enhanced.yaml`: the collaborative worker prompts. The enhanced one adds explicit position, next-agent, objective, and success-criteria fields and a submission workflow.
* `supervisor.yaml`, `supervisor_enhanced.yaml`: supervisor prompts. The enhanced one adds a workflow design framework and quality-gate guidance.
* `research_supervisor.yaml`: the prompt and configurable phases, file patterns, command patterns, and intervention triggers for `ResearchSupervisor`.
* `default_memory.yaml`: a worker prompt that uses MLGym's `memory_read` and `memory_write` tools to persist findings across steps.
* `default_lit_search.yaml`: a worker prompt that uses the `literature_search` tool.
* `default_gpu_info.yaml`, `human_in_loop.yaml`: GPU-aware and human-in-the-loop worker prompts.

## Tasks and benchmark

The task suite is MLGym's. The repository ships 18 task configs under `configs/tasks/` with their data and evaluation scripts under `data/`, spanning several domains:

* Computer vision: image classification (CIFAR-10, Fashion-MNIST), image captioning (COCO).
* Natural language: language modeling (FineWeb), natural language inference (MNLI).
* Reinforcement learning: Breakout (MinAtar), Meta Maze, Mountain Car Continuous.
* Game theory: Battle of the Sexes, Blotto, Prisoner's Dilemma.
* Regression: Kaggle house prices.
* Search and satisfiability: 3-SAT solving time.

Each task provides a baseline script and an `evaluate.py`. Agents must beat the baseline. The upstream MLGym-Bench v0 is described in the paper as 13 tasks; the configs here include those plus extra variants (for example L1 and alternate regression and RL formulations).

## Results

This repository does not report new quantitative results for the multi-agent system. There is no benchmark table, no aggregated supervisor scorecard, and no paper for the supervisor layer.

The trajectories checked into `trajectories/mlgym_bench_v0/` and their `results.json` files are upstream MLGym single-agent runs (models named `metagen-*` such as GPT-4o, o1, o3-mini, Claude 3.5 and 3.7 Sonnet, Gemini, DeepSeek, Llama). They belong to the base MLGym evaluation, not to the supervisor workflows added here. The supervisor environments do produce their own per-run `results.json` with per-agent and aggregated cost, token, and score fields when you run them, but no such runs are committed.

If you want numbers for the multi-agent system, you need to run it yourself and read the aggregated results file the supervisor environment writes.

## Repository structure

```
.
├── mlgym/                          # Core package (mostly upstream MLGym)
│   ├── agent/
│   │   ├── base.py                 # BaseAgent: the agent loop, format self-correction (upstream)
│   │   ├── supervisor_agent.py     # SupervisorAgent: plan/create/complete loop (added)
│   │   ├── research_supervisor.py  # ResearchSupervisor: scoring, phases, intervention (added)
│   │   ├── decoupled.py            # DecoupledAgent: propose without executing (added)
│   │   ├── supervisor_aware.py     # SupervisorAwareAgent: ask/submit-request worker (added)
│   │   ├── history_processors.py   # Memory windows: LastN/Last5/Last100Observations (upstream)
│   │   └── parsing.py              # Output parsers (upstream)
│   ├── environment/
│   │   ├── env.py                  # MLGymEnv, the base Gymnasium environment (upstream)
│   │   ├── supervisor_env.py       # SupervisorEnv: simpler multi-agent env (added)
│   │   ├── supervisor_env_mlgym.py # SupervisorEnvMLGym: main multi-agent env (added)
│   │   ├── tasks.py, registration.py, spaces.py (upstream)
│   ├── backend/                    # LiteLLM, human, and debugging backends (upstream)
│   ├── tools/                      # Tool definitions and command parsing (upstream)
│   ├── evaluation/                 # Scoring helpers (upstream)
│   ├── constants.py                # Supervisor/worker step limits and defaults (added)
│   └── types.py, exceptions.py, utils/ (upstream)
├── configs/
│   ├── agents/
│   │   ├── default*.yaml           # Single-agent worker prompts (default, memory, lit_search, gpu_info)
│   │   ├── supervisor*.yaml        # Supervisor prompts (basic and enhanced)
│   │   ├── research_supervisor.yaml
│   │   ├── worker_template*.yaml   # Collaborative worker prompts
│   │   ├── human_in_loop.yaml
│   │   └── orchestration/          # feature_eng.yaml, modeling.yaml, validation.yaml
│   ├── tasks/                      # 18 task configs (upstream suite plus variants)
│   └── datasets/                   # Dataset configs
├── data/                           # Per-task baselines, evaluate.py, sample submissions
├── tools/                          # Shell and Python tool implementations (upstream, SWE-agent derived)
├── dockerfiles/                    # Container images and aliases
├── demo/                           # Streamlit trajectory visualizer and demo
├── trajectories/mlgym_bench_v0/    # Upstream single-agent run trajectories and results
├── scripts/                        # Experiment shell scripts and result processing
├── notebooks/                      # Plotting
├── run.py                          # Upstream single-agent runner
├── run_supervisor_mlgym.py         # Main multi-agent runner (SupervisorEnvMLGym)
├── run_supervisor_basic.py         # Multi-agent runner (SupervisorEnv)
├── run_HIL.py                      # Human-in-the-loop runner (SupervisorAwareAgent)
├── multi_run.py                    # Parallel agents over GPUs
├── run_decoupled*.py               # Decoupled propose/verify drivers (see caveats)
├── run_sequential_experiment.py    # Sequential experiment driver
├── simple_run.py, single_experiment.py, run_replay.py (upstream-style helpers)
└── pyproject.toml
```

## Getting started

The environment setup is the same as upstream MLGym. Agents run inside a Docker or Podman container, and an LLM backend is reached through LiteLLM.

1. Clone and install.

   ```bash
   git clone https://github.com/limbajimba/Multi-agent-MLGym.git
   cd Multi-agent-MLGym
   conda create -y -n mlgym python=3.11
   conda activate mlgym
   pip install -e .
   ```

2. Create a `.env` file in the repository root with paths and API keys.

   ```bash
   MLGYM_CONFIG_ROOT="<path_to_repo>/configs"
   MLGYM_TASK_CONFIG_DIR="<path_to_repo>/configs/tasks"
   MLGYM_WORKSPACE_PATH="<path_to_repo>/workspace"
   MLGYM_ENV_TIMEOUT=10000
   MLGYM_ACTION_SHORT_TIMEOUT=60
   MLGYM_ACTION_LONG_TIMEOUT=10000
   MLGYM_MODEL_MAX_RETRIES=3

   OPENAI_API_KEY=""
   ANTHROPIC_API_KEY=""
   ```

3. Install Docker or Podman. Podman is the default container type in `mlgym/constants.py` and the recommended option on macOS. On Linux with GPUs, install the NVIDIA container toolkit. The upstream README has the full per-OS instructions, including Podman socket setup and CDI troubleshooting.

4. Pull the agent image.

   ```bash
   docker pull aigym/mlgym-agent:latest
   # or
   podman pull aigym/mlgym-agent:latest
   ```

## Running the multi-agent workflows

The main entry point is the MLGym-compliant supervisor runner. With no flags it defaults to the Battle of the Sexes task, a Podman container, and a small worker model.

```bash
python run_supervisor_mlgym.py
```

Common overrides:

```bash
python run_supervisor_mlgym.py \
  --environment.task_config_path tasks/regressionKaggleHousePrice.yaml \
  --environment.container_type docker \
  --supervisor_agent.model.model_name litellm:gpt-4o-mini \
  --supervisor_agent.agent_config_path configs/agents/supervisor_enhanced.yaml \
  --max_supervisor_steps 10 \
  --max_agents_per_workflow 5 \
  --max_steps_per_agent 30
```

The worker model and worker template are set in the script. By default workers use `configs/agents/worker_template_enhanced.yaml` and a small LiteLLM model. Edit `env.default_agent_args` in `run_supervisor_mlgym.py` to change the worker model or template.

Other runners:

```bash
# Simpler supervisor environment with decoupled workers
python run_supervisor_basic.py --environment.task_config_path tasks/battleOfSexes.yaml

# Human in the loop: you approve submissions and answer the agent's questions at the terminal
python run_HIL.py --environment.task_config_path tasks/regressionKaggleHousePrice.yaml

# Run several independent agents in parallel across GPUs
python multi_run.py --num_agents 4 --gpus_per_agent 1
```

Results and trajectories land under `trajectories/<user>/<run_name>/`. For supervisor runs, the supervisor's own trajectory is in a `supervisor/` subdirectory, each worker has its own `<role_n>/` subdirectory, and the aggregated `results.json` sits next to the supervisor trajectory.

## Single-agent MLGym (upstream)

The original single-agent flow still works unchanged. This runs one agent end to end on a task.

```bash
python run.py \
  --container_type docker \
  --task_config_path tasks/battleOfSexes.yaml \
  --model litellm:claude-3-5-sonnet-20240620 \
  --per_instance_cost_limit 4.00 \
  --agent_config_path configs/agents/default.yaml \
  --temp 1 \
  --gpus 0 \
  --max_steps 50 \
  --aliases_file ./dockerfiles/aliases.sh
```

Run `python run.py --help` for the full flag list.

## Trajectory visualizer

MLGym's Streamlit visualizer works for both single-agent and worker trajectories.

```bash
streamlit run demo/trajectory_visualizer.py -- --trajectory_dir <absolute_path_to_trajectories>
```

## Status and caveats

* Experimental. The upstream framework warns that MLGym is under heavy development. The multi-agent layer is research code added in one commit and should be read as such.
* The working multi-agent path is `run_supervisor_mlgym.py` with `SupervisorAgent` and `SupervisorEnvMLGym`, and the human-in-the-loop path is `run_HIL.py` with `SupervisorAwareAgent`. These import only modules that exist in the tree.
* The decoupled scripts are incomplete. `run_decoupled.py`, `run_decoupled_simple.py`, and `run_decoupled_supervisor.py` import `from mlgym.agent.supervisor import SupervisorAgent` and call `propose_action`, `review_action`, and `receive_feedback`. There is no `mlgym/agent/supervisor.py` in the tree (the supervisor class is in `supervisor_agent.py`), and those methods are not defined on any agent class. Treat these scripts as a sketch of a propose-and-review design, not as runnable code, unless you supply the missing module and methods. The `DecoupledAgent` class itself is present and used by `SupervisorEnv`.
* The README links a `CHANGELOG.md` and a `MAINTENANCE.md` that are inherited text from upstream and are not present in this repository.
* No multi-agent benchmark numbers are committed. See [Results](#results).

## Credit and citation

The base framework, the task suite, the tools, and most of the package are the work of GenAI at Meta and UCSB NLP. If you use MLGym, cite the original paper:

```tex
@misc{nathani2025mlgymnewframeworkbenchmark,
      title={MLGym: A New Framework and Benchmark for Advancing AI Research Agents},
      author={Deepak Nathani and Lovish Madaan and Nicholas Roberts and Nikolay Bashlykov and Ajay Menon and Vincent Moens and Amar Budhiraja and Despoina Magka and Vladislav Vorotilov and Gaurav Chaurasia and Dieuwke Hupkes and Ricardo Silveira Cabral and Tatiana Shavrina and Jakob Foerster and Yoram Bachrach and William Yang Wang and Roberta Raileanu},
      year={2025},
      eprint={2502.14499},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14499},
}
```

The tools under `tools/` are adapted from [SWE-agent](https://github.com/SWE-agent/SWE-agent), as noted in `tools/README.md`. Upstream project: [facebookresearch/MLGym](https://github.com/facebookresearch/MLGym).

## License

The majority of this code is licensed under CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International), inherited from upstream MLGym. Portions are under separate terms: SWE-agent and Modded-NanoGPT are MIT; Gymnax and Gymnax-blines are Apache 2.0. See `LICENSE`.
