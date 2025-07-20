"""
Arg-parser helper that (1) exposes selected config entries as CLI flags,
(2) writes overrides back into tinytransformer.config.config at runtime.
"""

import argparse
import importlib
from typing import Any, Callable

_CFG = importlib.import_module("tinytransformer.config.config")

# ------------------------------------------------------------------ #
# Helper to add a single flag with default from config module        #
# ------------------------------------------------------------------ #
def _add(parser: argparse.ArgumentParser, name: str, typ: Callable[[str], Any], desc: str):
    default = getattr(_CFG, name)
    parser.add_argument(f"--{name.lower()}", type=typ, default=default,
                        help=f"{desc} (default: {default})")

# ------------------------------------------------------------------ #
# Public: parse args & mutate config                                 #
# ------------------------------------------------------------------ #
def parse_and_apply() -> None:
    p = argparse.ArgumentParser("TinyTransformer CLI")
    # ░░ expose whatever you need ░░
    _add(p, "BATCH_SIZE",   int,   "mini-batch size")
    _add(p, "BLOCK_SIZE",   int,   "context / sequence length")
    _add(p, "LEARNING_RATE",float, "learning rate")
    _add(p, "NUM_STEPS",    int,   "training steps")
    _add(p, "D_MODEL",      int,   "hidden size")
    _add(p, "N_HEAD",       int,   "attention heads")
    _add(p, "NUM_LAYERS",   int,   "Transformer layers")
    _add(p, "TEMPERATURE",  float, "sampling temperature")
    _add(p, "TOP_P",        float, "nucleus top-p (0–1)")

    args = p.parse_args()

    # write back overrides
    for key, val in vars(args).items():
        setattr(_CFG, key.upper(), val)
