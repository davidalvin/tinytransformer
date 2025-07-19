import pytest
import torch
from tinytransformer.models.transformer import TinyTransformerLM
from tinytransformer.training.train import train, check_loss
from tinytransformer.training.generate import generate
from tinytransformer.config import config


@pytest.fixture
def small_model():
    return TinyTransformerLM()


def test_training_runs_without_crashing(small_model):
    # Run a very short training loop
    model = train(
        small_model,
        num_steps=2,
        batch_size=2,
        context_length=16,
        lr=1e-3,
        device="cpu"
    )
    assert isinstance(model, TinyTransformerLM)


def test_check_loss_returns_scalars(small_model):
    model = train(small_model, num_steps=1, batch_size=2, context_length=16, device="cpu")
    dummy_loss = torch.tensor(1.0)
    train_loss, val_loss = check_loss(0, dummy_loss, model, batch_size=2, context_length=16, device="cpu")
    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)


def test_generate_outputs_text(small_model):
    model = train(small_model, num_steps=1, batch_size=2, context_length=16, device="cpu")
    output = generate(model, "Once upon a time", max_tokens=5, context_length=16, device="cpu")
    assert isinstance(output, str)
    assert len(output) > 0
