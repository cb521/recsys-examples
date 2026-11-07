"""STUB - not ported yet: ``expert_placement``.

What it does: Expert re-placement across EP ranks from R3 routing counts: plans a permutation when imbalance is high and moves expert weights + optimizer state; probes and backs off when imbalance is low (auto-off).

Model family: MoE.  Frameworks: slime, verl.
Status, source location and what is left: docs/PORTING.md (section ``expert_placement``).
"""
from . import not_ported

NAME = "expert_placement"


def install(*args, **kwargs):
    not_ported(NAME)
