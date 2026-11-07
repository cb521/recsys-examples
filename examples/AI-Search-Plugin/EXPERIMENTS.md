# Experiments and ablations

_Work in progress. Numbers are added only after each comparison has been health-checked and independently reviewed._

## How we compare

- **Baseline = the strongest setting of the same framework**, tuned for the workload, with every speed-up the framework already ships turned on (e.g. its own async/partial-rollout options). Anything a baseline can get by changing its own config is credited to the baseline, not to the plugin.
- **Same node, same data, alternating order** (A→B and B→A), several steps, mean with confidence interval.
- **Both speed views**: seconds per step and seconds per million trained tokens. When the two sides train different numbers of tokens per step, we report both and say why.
- **Health checks on both sides**: entropy, train–inference log-prob gap, reward at start, tokens actually trained, max staleness.
- **Ablation**: leave-one-out from the full plugin set; one table row per component.

## Workloads

| Project | Model | Task | Frameworks compared |
|---|---|---|---|
| Search-R1 | Qwen2.5-7B (dense) | Multi-turn retrieval QA | original, verl, slime, AReaL, ROLL, plugin |
| ASearcher | Qwen2.5-7B (dense) | Multi-turn retrieval QA | original (AReaL), verl, slime, AReaL, ROLL, plugin |
| Libra | Qwen3-14B (dense) | Search-R1 task, short and long-tail settings | original, verl, slime, AReaL, ROLL, plugin |
| _Large sparse models_ | Qwen3-Next, GLM, DeepSeek-V3.2 class | _TBD_ | _TBD_ |

## Results

### Search-R1
_TBD_ — end-to-end speed (both views), accuracy (EM), component table.

### ASearcher
_TBD_

### Libra
_TBD_

## Component ablation table (template)

| Component | What it does | Full set | Without it | Gain | Baseline can get it? | Reviewed |
|---|---|---|---|---|---|---|
| | | | | | | |
