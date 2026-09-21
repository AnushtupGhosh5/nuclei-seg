"""Pinned official Mamba-UNet architecture source.

The implementation is derived from ziyangwang007/Mamba-UNet commit
2eeec299581934e05b2af0322cc3107e2605867a. See LICENSE in this directory.
Only compatibility imports and optional profiling imports differ.
"""

from .mamba_sys import VSSM

__all__ = ["VSSM"]
