import torch
import torch.nn as nn

from tinytransformer.models.base_lm import LanguageModel
from tinytransformer.config import config as C


class TinyTransformerLM(LanguageModel):
    def __init__(
        self,
        vocab_size=C.VOCAB_SIZE,
        d_model=C.D_MODEL,
        nhead=C.N_HEAD,
        num_layers=C.NUM_LAYERS,
        max_seq_len=C.MAX_SEQ_LEN,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head     = nn.Linear(d_model, vocab_size)

        # optional: weight-tying (helps small LMs)
        self.lm_head.weight = self.token_embed.weight

    # ---------- forward ---------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        T = min(T, self.pos_embed.size(1))
        x = self.token_embed(x) + self.pos_embed[:, :T]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        x = self.transformer(x, mask=mask)
        return self.lm_head(x)
