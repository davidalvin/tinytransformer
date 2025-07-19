from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from tinytransformer.data.data import get_batch                          # <-- Fixed: import from data package
from tinytransformer.models.transformer import TinyTransformerLM        # <-- Fixed: use correct module name
from tinytransformer.config.config import NUM_STEPS, BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE # use shared config
      
def check_loss(step, loss, model, batch_size, context_length, device):
    model.eval()
    with torch.no_grad():
        X_val, Y_val = get_batch("val", batch_size, context_length, device=device)
        val_logits = model(X_val).flatten(0, 1)
        val_loss = F.cross_entropy(val_logits, Y_val.flatten(0, 1))
    model.train()
    
    return loss.item(), val_loss.item()


def train(
    model,
    num_steps=NUM_STEPS,
    batch_size=BATCH_SIZE,
    context_length=BLOCK_SIZE,
    lr=LEARNING_RATE,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    print(f"📦 Model on device: {device}")

    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"Starting training for {num_steps}")
    with trange(num_steps, desc="🧠 training", unit="step") as pbar:
        for step in pbar:
            # X:        (B, T)
            # logits:   (B, T, vocab_size), note B = batch size and T = context length
            # Y:        (B, T)
            X, Y = get_batch("train", batch_size, context_length, device=device)  # <-- Fixed: add split parameter
            
            logits = model(X)

            # Flatten to compute cross-entropy over (B*T, vocab_size)
            logits_flat = logits.flatten(0, 1)
            Y_flat = Y.flatten(0, 1)

            loss = F.cross_entropy(logits_flat, Y_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0 or step == num_steps - 1:
                train_loss, val_loss = check_loss(step, loss, model, batch_size, context_length, device)
                tqdm.write(f"step {step:>4} / {num_steps}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
                pbar.set_postfix(train=train_loss, val=val_loss)

    return model