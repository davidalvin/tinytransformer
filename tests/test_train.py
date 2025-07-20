import pytest
import torch
from tinytransformer.models.transformer import TinyTransformerLM
from tinytransformer.training.train import train, check_loss
from tinytransformer.models.factory import build_model
from tinytransformer.data.data import get_batch  


@pytest.fixture
def small_model():
    return TinyTransformerLM()


def test_training_runs_without_crashing(small_model):
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

    # Just validate that check_loss returns a scalar
    val_loss = check_loss(model, batch_size=2, context_length=16, device="cpu", get_batch_fn=get_batch)
    assert isinstance(val_loss, float)


def test_generate_outputs_text(small_model):
    model = train(small_model, num_steps=1, batch_size=2, context_length=16, device="cpu")
    output = model.generate("Once upon a time", max_tokens=5, context_length=16, device="cpu")
    assert isinstance(output, str)
    assert len(output) > 0

def test_gradients_flow():
    model = build_model("tiny")
    model.train()

    x = torch.randint(0, model.token_embed.num_embeddings, (2, 8))
    y = torch.randint(0, model.token_embed.num_embeddings, (2, 8))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "No gradients flowed"