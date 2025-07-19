# tests/test_tokenizer.py
"""
Unit tests for the TinyTransformer tokenizer helpers.

Assumes you have:

    from my_package.tokenizer import encode, decode

Replace the import line below with the actual module path.
"""

import re
import pytest

# ─── Adjust this import to your project layout ────────────────────────────────
from tinytransformer.models.tokenizer import encode, decode   # ← change if needed


@pytest.mark.parametrize(
    "text",
    [
        "Hello world!",
        "Once upon a time there was a little girl named Lily.",
        "  Leading and trailing  spaces   ",
    ],
)
def test_round_trip_identity(text: str):
    """
    Encoding followed by decoding should return the original string
    (modulo leading/trailing space cleanup from `clean_up_tokenization_spaces=True`).
    """
    ids = encode(text)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids), "encode() didn't return List[int]"

    recovered = decode(ids)

    # clean_up_tokenization_spaces may collapse excess whitespace;
    # so compare after normalizing internal whitespace.
    normalize = lambda s: re.sub(r"\s+", " ", s.strip())
    assert normalize(recovered) == normalize(text), f"Mismatch:\norig: {text!r}\nrec : {recovered!r}"


def test_no_bytelevel_artifacts():
    """
    Decoded text should not contain the GPT-2 byte-level space marker 'Ġ'
    or the Unicode replacement char '�'.
    """
    bad_chars = {"Ġ", "�"}
    sample    = "The quick brown fox jumps over the lazy dog."

    ids  = encode(sample)
    text = decode(ids)

    assert not any(c in text for c in bad_chars), f"Found byte-level artifacts in decoded text: {text!r}"


def test_deterministic_encoding():
    """
    Calling encode() twice on the same string should yield identical ID sequences.
    """
    text = "Determinism test."
    ids1 = encode(text)
    ids2 = encode(text)
    assert ids1 == ids2, "encode() is not deterministic for identical input"


def test_special_token_strip():
    """
    If the tokenizer inserts special tokens like <|endoftext|>, ensure decode() removes them.
    """
    # fabricate a sequence with a known special-token id if you know it,
    # else just round-trip a sentence and ensure the token isn't in the output.
    ids = encode("Testing special token handling.") + encode("<|endoftext|>")
    decoded = decode(ids)
    assert "<|endoftext|>" not in decoded, "Special token leaked into decode() output"
