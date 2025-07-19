I'm working through the content of CS336 (Stanford’s “Language Models” course) on my own as a hobby project. The goal is to deeply understand and implement a Transformer-based language model from the ground up — not just use prebuilt libraries.

I'm following the structure of Assignment 1, which walks through:

Tokenization (UTF-8 → eventually BPE)

Transformer architecture

Training with AdamW

Inference and text generation

But since I'm not enrolled in the class and I'm doing this for personal learning, I'm not strictly following all the course constraints (e.g., I may use torch.nn.Linear early on instead of reimplementing it immediately). The point is understanding, not rule-following.

What I’m Trying to Learn:

How each component of a Transformer language model works

How to train a model on raw text, from tokenization to sampling

What trade-offs real LLM engineers think about (e.g., byte vs BPE, norm layers, etc.)

How to debug, optimize, and interpret training dynamics

My Background:
I’m a self-taught developer / engineer with limited experience in deep learning. I’m comfortable with Python, PyTorch basics, and general software engineering — but I’m here to build real intuition about deep language models by doing the hard parts myself.

This is not a plug-and-play “use transformers library” project. It’s a low-level, from-scratch implementation to understand the system end to end.

How I’m Approaching It:

Start with simple, working prototypes (e.g., byte-level tokenizer, tiny Transformer on TinyStories)

Gradually replace PyTorch built-ins with manual implementations

Write modular code with tests for each stage (tokenizer, dataset, model, trainer)

Keep flexibility to break the CS336 rules if it speeds up learning


FULL COURSE REFERENCE BUT NOTE THE CAVEATS ABOVE
CS336 ASSIGNMENT 1 - FULL REFERENCE
SUMMARY
Build a Transformer Language Model (LM) from scratch.
Do NOT use torch.nn.Linear, torch.nn.functional, or torch.optim.
You CAN use: torch.nn.Parameter, torch.nn.Module, torch.optim.Optimizer.

Components:

BPE tokenizer

Transformer LM (attention, feedforward, etc.)

Cross-entropy loss, AdamW optimizer

Training loop (dataloader, checkpointing, scheduler)

Text generation (greedy, temperature, top-p sampling)

Ablations and experiments (TinyStories, OpenWebText)

PROJECT STRUCTURE
cs336_basics/ <- Your code goes here
adapters.py <- Hook functions used in test scripts
tests/test_*.py <- Do NOT edit; run to check correctness

You will implement: Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention, TransformerBlock, etc.

BYTE-PAIR ENCODING (BPE) TOKENIZER
Tokenizer training function signature:
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]

Steps:

Convert text to UTF-8 bytes

Pre-tokenize using regex:
PATTERN: r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Initialize vocab: 256 byte values + special tokens

Count byte-pairs within pre-tokens (not across them)

Repeatedly merge most frequent pairs until vocab_size reached

Deterministic tie-breaking: use max(pair)

Tokenizer class interface:
class Tokenizer:
def init(self, vocab, merges, special_tokens=None)
@classmethod
def from_files(cls, vocab_path, merges_path, special_tokens=None)
def encode(self, text: str) -> list[int]
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
def decode(self, ids: list[int]) -> str

TRANSFORMER ARCHITECTURE
BUILDING BLOCKS (subclass torch.nn.Module):
Linear:

No bias

Input: (..., in_features)

Output: (..., out_features)

Weight shape: (out_features, in_features)

Init: N(0, 2 / (in + out)), truncated at ±3σ

Embedding:

Maps token ID to vector

Shape: (num_embeddings, embedding_dim)

Init: N(0, 1), truncated at ±3

RMSNorm:

Normalize over last dim

Formula: x / RMS(x) * g

g is learnable gain, shape (d_model,)

RMS = sqrt(mean(x^2) + eps), eps = 1e-5

SwiGLU (FeedForward):

d_ff = round(8/3 * d_model), multiple of 64

Output: W2(SiLU(W1x) * W3x)

Use torch.sigmoid or SiLU as needed

RoPE (Rotary Positional Embedding):

For each token pos i, rotate embedding pairs using cos/sin

θ = position-dependent rotation

Apply to Q and K, not V

Cache sin/cos using register_buffer(persistent=False)

Scaled Dot Product Attention:

Q, K, V → Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Support boolean mask M: shape (seq, seq), False = mask out

Multi-Head Self-Attention:

Parameters: W_q, W_k, W_v, W_o

d_head = d_model / num_heads

RoPE applied to Q, K

Causal mask: prevent looking ahead

TransformerBlock:

Pre-norm

Layer 1: RMS → MHA → residual

Layer 2: RMS → FFN → residual

TransformerLM:

Embedding

N TransformerBlocks

Final RMSNorm

Output projection (Linear → vocab_size)

TRAINING
Loss: Cross Entropy

Softmax + log + NLL over predicted logits

Use max(logits) subtraction for stability

Return mean over all tokens and batch

AdamW Optimizer:

Params: lr, betas, eps, weight_decay

Track m, v, t per param

Update:
m = β1 * m + (1 - β1) * g
v = β2 * v + (1 - β2) * g²
α_t = lr * sqrt(1 - β2^t) / (1 - β1^t)
param -= α_t * m / (sqrt(v) + eps)
param -= lr * weight_decay * param

Gradient Clipping:

If ||grad||₂ > max_norm, scale down by max_norm / (||grad||₂ + eps)

Learning Rate Scheduler (Cosine w/ Warmup):

warmup_steps = Tw

decay_steps = Tc

schedule:
if t < Tw:
lr = (t / Tw) * lr_max
elif t < Tc:
lr = lr_min + 0.5 * (1 + cos(pi * (t-Tw)/(Tc-Tw))) * (lr_max - lr_min)
else:
lr = lr_min

Checkpointing:

Save: torch.save({'model': model.state_dict(), 'optim': optim.state_dict(), 'step': step}, path)

Load: torch.load(), then load_state_dict

Data Loader:

Tokenized data: numpy array of dtype=uint16

Input: x[i : i+ctx_len], Target: x[i+1 : i+ctx_len+1]

Use np.memmap for large files

TEXT GENERATION
Decode loop:

While <|endoftext|> not seen and max tokens not reached:

Feed prompt to model

Take logits at last position

Apply temperature: logits /= temp

Optional: top-p (nucleus) sampling

Sample next token → append → repeat

EXPERIMENTS AND HYPERPARAMETERS
TinyStories Base Config:

vocab_size: 10000

context_length: 256

d_model: 512

d_ff: 1344

num_layers: 4

num_heads: 16

RoPE theta: 10000

total tokens seen: 327M

Log:

Validation loss over time (steps, wall time)

Use Weights & Biases (optional)

Ablation Ideas:

Remove RMSNorm → unstable?

Post-norm vs Pre-norm

NoPE (no positional embedding)

SwiGLU vs SiLU (non-gated FFN)

Large-Scale:

OpenWebText dataset

32K vocab

Same model, longer training

Leaderboard: best validation loss in ≤ 1.5 hours on H100

TESTING
Run tests:

uv run pytest -k test_name

Hook via adapters.py:

run_linear()

run_embedding()

run_rmsnorm()

run_swiglu()

run_scaled_dot_product_attention()

run_transformer_block()

run_transformer_lm()

run_cross_entropy()

run_gradient_clipping()

run_get_batch()

run_save_checkpoint()

run_load_checkpoint()

get_adamw_cls()

get_tokenizer()

get_lr_cosine_schedule()