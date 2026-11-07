"""STUB - not ported yet: ``hybridep_overlap``.

What it does: HybridEP dispatch/combine with a 2-chunk compute/communication overlap in the MoE layer.

Model family: MoE.  Frameworks: slime.
Status, source location and what is left: docs/PORTING.md (section ``hybridep_overlap``).
"""
from . import not_ported

NAME = "hybridep_overlap"


def install(*args, **kwargs):
    not_ported(NAME)
