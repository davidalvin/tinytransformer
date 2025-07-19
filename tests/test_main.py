"""
Pytest that runs the full `main()` for ONE step on dummy data.
Nothing touches the real dataset; nothing is written outside a tmp dir.
"""

import os
import torch
from tinytransformer import main


def dummy_get_batch(split, batch_size, context_length, device):
    """Return random ints in the model's vocab range (default 2048)."""
    vocab_size = 2048
    X = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    Y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return X, Y


def test_main_runs_one_step(tmp_path):
    # Redirect model save to a tmp file
    save_path = tmp_path / "tiny.pt"

    # Execute main() for a single training step on CPU with dummy batches
    main.main(
        num_steps=1,
        batch_size=2,
        context_length=8,
        lr=1e-3,
        device="cpu",
        get_batch_fn=dummy_get_batch,
        model_path=str(save_path),
    )

    # Assert model file was created
    assert save_path.exists(), "Model file was not written"
