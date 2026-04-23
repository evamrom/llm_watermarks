"""
watermark.py — Undetectable LLM Watermarking (Christ, Gunn, Zamir 2023)

Implements the watermarking algorithm from Section 4.3 (Algorithm 3) of the paper,
adapted for GPT-2's full vocabulary using the inverse-CDF trick described in Section 4.2.

Key idea:
  - Generate tokens naturally while accumulating empirical entropy.
  - Once entropy >= lambda (threshold), lock the token prefix as seed r.
  - For all subsequent tokens: u = F_sk(r, t) via HMAC-SHA256 (PRF),
    then select token via inverse CDF on prob-sorted vocabulary.
  - Output distribution is UNCHANGED — watermark is undetectable without key.
"""

import hmac
import hashlib
import math
import struct
from typing import Optional
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_KEY = b"spc_feh_2026_watermark_key"
DEFAULT_LAMBDA = 4.0          # λ: entropy threshold for seed locking (generation) and detection margin
# ──────────────────────────────────────────────────────────────────────────────


def prf(key: bytes, seed_tokens: list[int], position: int) -> float:
    """
    Pseudorandom function F_sk(r, pos) → [0, 1].

    Uses HMAC-SHA256(key, encode(r) || encode(t)). The first 8 bytes of the
    digest are interpreted as a 64-bit integer and normalised to [0, 1].
    This replaces the random oracle O used in Section 3 of the paper.
    """
    seed_bytes = b"".join(struct.pack(">I", tok) for tok in seed_tokens)
    pos_bytes = struct.pack(">I", position)
    digest = hmac.new(key, seed_bytes + pos_bytes, hashlib.sha256).digest()
    val = int.from_bytes(digest[:8], "big")
    return val / (2 ** 64)


def inverse_cdf_sample(probs: torch.Tensor, u: float) -> int:
    """
    Select a token by placing u on the CDF number line built from sorted probs.

    Algorithm:
      1. Sort tokens by probability (descending) → forms a number line 0 → 1.
      2. Find the first token whose cumulative probability reaches u.
      3. Return that token's id.

    Because u ~ Uniform[0,1] (to anyone without the key), the marginal
    distribution of the selected token equals the original probs — this
    is the core undetectability argument (Section 4.2 of the paper).
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs.double(), dim=0)

    # Clamp to avoid numerical edge cases at 0 and 1
    u_clamped = max(min(u, 1.0 - 1e-9), 1e-9)
    idx = int(torch.searchsorted(cumsum, torch.tensor(u_clamped, dtype=torch.float64)))
    idx = min(idx, len(sorted_indices) - 1)
    return int(sorted_indices[idx])


class WatermarkGenerator:
    """
    Generates watermarked text on top of GPT-2.

    Phase 1 (natural sampling): tokens are drawn from the model's distribution
    while the cumulative empirical entropy is tracked.

    Phase 2 (PRF-guided sampling): once entropy >= lambda, the token prefix is
    locked as seed r. Every subsequent token is selected via inverse-CDF with
    u = F_sk(r, t), preserving the output distribution exactly.
    """

    def __init__(
        self,
        key: bytes = DEFAULT_KEY,
        lambda_: float = DEFAULT_LAMBDA,
        model_name: str = "gpt2",
        model: Optional[GPT2LMHeadModel] = None,
        tokenizer: Optional[GPT2Tokenizer] = None,
    ):
        self.key = key
        self.lambda_ = lambda_
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            print(f"Loading {model_name}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> dict:
        """
        Generate watermarked text for a given prompt.

        Returns a dict with:
          - 'text':             decoded generated text (without prompt)
          - 'generated_tokens': token ids of generated text only
          - 'seed':             locked seed token list (or None if entropy never crossed λ)
          - 'seed_position':    number of tokens in the seed (ℓ from the paper)
          - 'entropy_reached':  final cumulative entropy in nats
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        cumulative_entropy = 0.0
        seed: Optional[list] = None
        generated: list[int] = []

        # ── Initial forward pass on the full prompt (builds KV cache) ────────
        outputs = self.model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

        for step in range(max_new_tokens):
            if seed is None:
                # ── Phase 1: natural sampling + entropy accumulation ─────────
                token = int(torch.multinomial(probs, 1))
                token_prob = float(probs[token])
                cumulative_entropy += -math.log(token_prob + 1e-15)
                generated.append(token)
                if cumulative_entropy >= self.lambda_:
                    seed = list(generated)
            else:
                # ── Phase 2: PRF-guided sampling (inverse CDF) ───────────────
                pos = len(generated)
                u = prf(self.key, seed, pos)
                token = inverse_cdf_sample(probs, u)
                generated.append(token)

            if token == self.tokenizer.eos_token_id:
                break

            # ── Next step: only process the new token (O(1) via KV cache) ────
            outputs = self.model(
                torch.tensor([[token]]),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return {
            "text": text,
            "generated_tokens": generated,
            "seed": seed,
            "seed_position": len(seed) if seed else None,
            "entropy_reached": cumulative_entropy,
        }


if __name__ == "__main__":
    gen = WatermarkGenerator()
    prompt = "The study of cryptography has a long history"
    print(f"Prompt: {prompt}\n")

    result = gen.generate(prompt, max_new_tokens=80)
    print(f"Generated text:\n{result['text']}\n")
    print(f"Entropy at seed lock: {result['entropy_reached']:.2f} bits")
    print(f"Seed locked at token position: {result['seed_position']}")
    print(f"Seed length (tokens): {len(result['seed']) if result['seed'] else 0}")
