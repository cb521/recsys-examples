"""STUB - not ported yet: ``r3_ep_plan``.

What it does: R3-budgeted EP plan: with rollout routing replay the routes of a micro-batch are known before the forward, so the EP all-to-all sizes / buffers are planned ahead (sync-free dispatch, chunked), with a runtime guard.

Model family: MoE with rollout routing replay (R3).  Frameworks: slime.
Status, source location and what is left: docs/PORTING.md (section ``r3_ep_plan``).
"""
from . import not_ported

NAME = "r3_ep_plan"


def install(*args, **kwargs):
    not_ported(NAME)
