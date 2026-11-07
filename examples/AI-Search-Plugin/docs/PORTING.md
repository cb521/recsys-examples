# Porting notes

Where each component came from, what changed while porting it, and what is left for the stubs.
Source = the internal development repository of the AI-search RL project (not public); paths below are relative to
that repository's root, branches are its branch names.

## Ported

Source of almost everything: branch `exp/e2e-phase4-20261026`, package `e2e/phase4/asrt/python/asrt/`
(the "asrt" control-plane package), at commit `e52cae79`.  `response_logits`: branch `exp/train-perf-r10-chunkce`,
`components/response_logits/`, at commit `9705d122`.

| plugin module | source module |
|---|---|
| `adapters/_hooks.py`, `common.py`, `staleness_core.py`, `slime_*.py`, `tool_hedge.py`, `jitcache.py`, `verl*.py`, `sgl_boot/sitecustomize.py` | `asrt/adapters/<same name>` |
| `toolcache_py.py`, `partition_py.py`, `r3pack.py` | `asrt/<same name>` |
| `response_logits/` (`__init__`, `chunked`, `adapters/{slime,verl}`) | `components/response_logits/` |
| `tests/cpu/*`, `tests/scripts/*` | `e2e/phase4/asrt/tests/test_<name>.py` (script-style tests renamed, see below) |
| `tests/integration/slime_rshard_ray.py`, `rshard_fake_agent.py` | `e2e/phase4/asrt/tests/test_slime_rshard_ray.py`, `rshard_fake_agent.py` |
| `tests/gpu/*` | `components/response_logits/tests/test_{chunked_op,adapters_gpu}.py` |

What changed in the port (behaviour of the shipped components is otherwise unchanged):

1. **Package paths**: `asrt.adapters.X` -> `ai_search_plugin.adapters.X`, `response_logits` ->
   `ai_search_plugin.response_logits`.  Env variable names (`ASRT_*`) and log prefixes (`[asrt]`) are kept so existing
   job scripts and log parsers keep working.  `ai_search_plugin.adapters.verl_plugin` stays as an alias of
   `ai_search_plugin.verl_plugin`.
2. **No C++ extension**: asrt could use a C++ tool cache / partitioner / router; the plugin ships only the pure-Python
   tool cache and partitioner (`toolcache_py`, `partition_py`).  `ExpertPlacementHook` and the C++ cost router
   (`verl.pick_server`) were dropped with it (expert placement is a stub; the verl `route` feature uses its own
   pure-Python picker).
3. **Deprecated components removed** (tried, no measured gain): out-of-order batch assembly (`ASRT_STALE_MODE=ooo`,
   `OooTickets` class), straggler slip (`ASRT_TICKET_SLIP`, now a no-op), slow-tool cut (`ASRT_TICKET_SLOWTOOL_S`,
   module `slowtool` not shipped), closed-loop extra fill (`ASRT_STALE_FILL`, forced off), verl-window cut
   (`ASRT_TICKET_DROP=verl4`, forced off - its branch in `Tickets._drop_overdue` is dead code left for a later cleanup),
   elastic GPU lending (`slime_elastic`, `verl_elastic`; the dormant "lent engine" code paths in `slime_rshard` and the
   `lender` hooks in `slime_train` remain but are never activated), cross-engine rebalancing (`slime_rbal`), expert
   replication.  `ai_search_plugin.install()`, the slime init hook and the verl plugin refuse these switches
   (`registry.DEPRECATED_ENV`).  Their tests were not ported.
4. **The train-perf bundle (`asrt.adapters.trainperf`) is not shipped** (its model-specific kernel selection belongs
   to one proprietary model).  In `adapters/verl.py` the `trainperf` feature was replaced by a `response_logits`
   feature that installs only the ported response_logits verl adapter + its guard; the opt-in FSDP response_logits hook
   was dropped (its adapter is not in the source branch).
5. **Defaults that named an internal agent were removed**: `slime_toolcache` / `tool_hedge` had default tool targets
   pointing at the project's own search agent; now the user must list them (`ASRT_TOOLCACHE_GENERATE`,
   `ASRT_TOOLCACHE_TARGETS`, `ASRT_TOOL_HEDGE_TARGETS`, or the `toolcache_generate` / `tool_targets` options).
6. **Comments/docstrings sanitized**: cluster names, job ids, internal file paths, run ids, review hashes and
   model-specific numbers of a proprietary model were removed or replaced by neutral wording; internal milestone
   tags were stripped.  The engineering explanation was kept.
7. `slime_weightsync.megatron_init` no longer starts the `stallwatch` diagnostic (module not shipped).
8. **Tests**:
   - script-style tests (module-level asserts / `__main__` runners) live in `tests/scripts/` without the `test_`
     prefix and are run each in a fresh interpreter by `tests/test_ported_scripts.py`; plain pytest files are in
     `tests/cpu/`.
   - `slime_admit_cpu.py` / `slime_admit_resume_cpu.py` load slime's real `rollout/fully_async_rollout.py`; they used
     a hard-coded internal copy and now read `$SLIME_SRC` (or an installed slime) and are skipped otherwise.
   - `cpu/test_slime_bigbucket_cpu.py`: two tests failed in the source repo too (the fake engine lacked the
     `_actor_id` attribute Ray handles have); the fixture now has it.  No library change.
   - `slime_admit_cpu.py::test_flush1_cut_requeue_e2e`: whether the cut+requeue path is hit is timing dependent (the
     source test also missed it in about 1 of 4 isolated runs).  It now retries up to 4 times for path coverage; every
     attempt still checks all correctness assertions.
   - tests of removed features were dropped (ooo / slip / slow-tool / verl4 / placement / the internal kvprobe
     converter); `verl_rlogits_guard_cpu.py` and `verl_adapter_cpu.py` fake `ai_search_plugin.response_logits`
     instead of a top-level `response_logits` package; tool-hedge tests use explicit targets.
   - **not run here**: `tests/gpu/*` (needs CUDA + Megatron, `torchrun`) and `tests/integration/slime_rshard_ray.py`
     (needs slime + Ray + a tokenizer).

Present in the ported code but **not registered** (benefit undecided, kept off): verl `route`, `keepkv` (as a
stand-alone feature; `verl_inplace` turns it on), `tokcache`, `balance`, `profile`, `offpol` (measurement only, used
by `verl_ceiling`'s tests).  They can be enabled by hand through `ASRT_VERL=<list>`; registering one = adding a
`Component(...)` entry.

## Stubs (not ported)

Each has a registry entry and a module in `src/ai_search_plugin/stubs/` whose `install()` raises `NotPortedError`.

### `prefix_tree` (model-agnostic, Megatron backend)
- Source: branch `exp/innov1-prefix-tree-20261102`, `components/prefix_tree/` (training-side tree packing,
  `verl_driver` keeps a GRPO group on one DP rank, `verl_adapter`), wired through `components/train_perf_plugins/`.
- Left: port the packing + attention-mask / position-id construction for Megatron; slime adapter and verl driver +
  adapter; GPU equivalence tests (loss / grads vs the flat path).  Open item in the source: the root cause of a
  measured per-token log-prob difference vs the flat path on one dense model is not established - verify before
  shipping.

### `r3_ep_plan` (MoE with R3)
- Source: branch `exp/innov2-r3plan-20261102`, `components/ep_dispatch/r3_plan_a2a.py` (+ `overlap.py`,
  `recompute_cache.py`, `test_r3plan.py`), driven from `scripts/hybrid_train.py` (`TP_MOE_R3_PLAN`).
- Left: make it a Megatron MoE dispatcher patch usable from slime (the source runs it inside its own hybrid trainer;
  slime's all-to-all path is not default there); runtime guard (per-step plan-vs-actual mismatch check) must come
  with it.  Validated layouts in the source: PP1, PP2/VPP2 (smoke), PP4/VPP3; other layouts rely on the guard.

### `mtp_draft_loader` (models with an MTP head)
- Source: branch `exp/e2e-phase4-20261026`, slime launch configuration (not code in the asrt package): SGLang
  `--speculative-algorithm NEXTN` with fixed steps + `--sglang-speculative-draft-load-format auto` so the draft head
  follows weight syncs, plus the R3 audit (`ASRT_R3AUDIT`, already in `slime_r3check`).
- Left: decide whether this is a component or plain framework configuration (the source classifies it both ways);
  if a component, emit the SGLang arguments from `frameworks/slime.py` and document the draft-head sync.

### `dsa_replay` (DSA sparse attention)
- Source: branch `exp/e2e-phase4-20261026`, `e2e/phase4/asrt/python/asrt/dsa_replay/` (`codec`, `store`,
  `rollout`, `slime_rollout`, `slime_train`) with CPU tests `test_dsa_replay_cpu.py`, `test_dsa_replay_hook_cpu.py`.
- Left: the rollout-side capture needs SGLang to expose the per-token top-k selection (not available in the tested
  SGLang); never run end to end.  The codec/store are pure Python and could be ported first.

### `expert_placement` (MoE)
- Source: branch `exp/e2e-phase4-20261026`, `asrt/adapters/megatron_placement.py`, `slime_placement.py`,
  `verl_placement.py`, `place_verify.py`, `asrt/placement_py.py`; CPU tests `test_placement_cpu.py`,
  `test_place_*_cpu.py`, `test_slime_placement_cpu.py`.
- Left: the planner used the C++ extension (`solvers.plan_placement`); needs the pure-Python planner wired in
  (`placement_py` has `imbalance` / `rebalance_swaps`).  Auto-off when imbalance is low must be kept.

### `hybridep_overlap` (MoE)
- Source: train-perf line, `components/ep_dispatch/` (`--tp-moe-dispatcher hybridep --tp-moe-ep-overlap 2`);
  HybridEP itself is Megatron's flex dispatcher backend (needs a Megatron overlay build in the source setup).
- Left: express it as Megatron arguments + the 2-chunk overlap patch for slime; not wired for verl.

### `gdn_cudnn` (GDN; slime only)
- Source: branch `exp/train-perf-r10-merge`, `kernels/gdn_cudnn/`; loader overlay `asrt/adapters/gdn_overlay.py`
  (+ CPU test `test_gdn_overlay_cpu.py`); selection was inside the train-perf bundle.
- Left: port the kernel binding + overlay, the slime install hook, and the jitcache `cudnn` store integration
  (already in `adapters/jitcache.py`).  The verl use is deprecated (no gain, more memory).

### `fla_nb_bucket` (FLA / GDN)
- Source: branch `exp/train-perf-r10-merge`, `kernels/fla_nb_bucket/` and the `ASRT_FLA_NB_PIN` pinning in
  `asrt/adapters/trainperf.py` (`_pin_nb`).
- Left: stand-alone install hook for slime and verl (it was part of the train-perf bundle).

### `gdn_convfuse_ep` (GDN / MoE)
- Source: branch `exp/train-perf-r10-merge`, `kernels/gdn_conv_fuse/` and the round-10 EP items
  (`components/ep_dispatch/`).
- Left: these were tuned on one model's 32-GPU layout; re-validate on an open GDN/MoE model before porting.

## Not part of the plugin

- Sparse-attention kernels and other components that only fit one proprietary model's structure.
- Framework tuning that the baseline can enable itself (SGLang prefill graphs, concurrency / prefetch factor, bucket
  size, recompute / PP overlap settings, NCCL CTA count).
- The deprecated components listed above.
