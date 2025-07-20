from tinytransformer.models.transformer import TinyTransformerLM
from tinytransformer.models.base_lm import LanguageModel


def build_model(name: str = "tiny", **kwargs) -> LanguageModel:
    if name == "tiny":
        return TinyTransformerLM(**kwargs)
    raise ValueError(f"Unknown model: {name}")
