import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tinytransformer.models.tokenizer import encode, decode


class LanguageModel(ABC, nn.Module):
    """Abstract base LM with reusable helpers."""

    # ---------- required ---------- #
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    # ---------- default generation ---------- #
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        context_length: int = 256,
        device: str = "cpu",
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> str:
        self.eval()
        self.to(device)

        x = self._encode_prompt(prompt, context_length, device)
        for _ in range(max_tokens):
            logits = self(x[:, -context_length:])        # (1, T, vocab)
            next_id = self._sample_next_token(
                logits[0, -1], temperature=temperature, top_p=top_p
            )
            x = torch.cat([x, next_id.unsqueeze(0)], dim=1)

        return decode(x[0].tolist())

    # ---------- helpers ---------- #
    def _encode_prompt(self, prompt: str, context_length: int, device: str) -> torch.Tensor:
        tokens = encode(prompt)[-context_length:] or [0]   # guard empty
        return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> torch.Tensor:
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        if top_p is not None and 0 < top_p < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            sorted_probs[mask] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum()

        return torch.multinomial(probs, num_samples=1)
