"""STUB - not ported yet: ``prefix_tree``.

What it does: Prefix-tree training of GRPO groups: the samples of one group share the prompt, so the shared prefix is packed once as a tree (custom attention mask / position ids) and the loss and gradients equal the flat path up to bf16 noise; the driver keeps siblings on one DP rank.

Model family: model-agnostic (Megatron backend).  Frameworks: slime, verl.
Status, source location and what is left: docs/PORTING.md (section ``prefix_tree``).
"""
from . import not_ported

NAME = "prefix_tree"


def install(*args, **kwargs):
    not_ported(NAME)
