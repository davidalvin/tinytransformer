import torch
import torch.nn as nn

from tinytransformer.config.config import (
    VOCAB_SIZE,
    D_MODEL,
    N_HEAD,
    NUM_LAYERS,
    MAX_SEQ_LEN,
)


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    ):
        super().__init__()

        # Embed token IDs into d_model-dimensional vectors
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Learnable positional embeddings (max length: MAX_SEQ_LEN)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Single Transformer encoder layer (reused for stacking)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True  # input shape is (B, T, d_model)
        )

        # Stacked Transformer layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection to vocab size (the "LM head")
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, T) — raw token IDs
        B, T = x.size()

        # Add token + positional embeddings → shape: (B, T, d_model)
        x = self.token_embed(x) + self.pos_embed[:, :T, :]

        # Causal mask: (T, T), upper triangle is -inf to prevent attending to future tokens
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).to(x.device)

        # Run through Transformer layers → (B, T, d_model)
        x = self.transformer(x, mask=mask)

        # Project to vocabulary logits → (B, T, vocab_size)
        return self.lm_head(x)
