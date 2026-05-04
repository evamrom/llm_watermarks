"""
detect.py — Watermark Detector (Christ, Gunn, Zamir 2023)

Implements the model-free detection algorithm from Section 4.3 (Algorithm 4).

Key breakthrough of the binary alphabet reduction:
  The detector DOES NOT NEED THE LANGUAGE MODEL. Because the generator evaluated
  p(1) bit-by-bit and chose bit = 1 if u <= p(1), the mathematical expectation 
  is such that if a text is watermarked, the actual bit x_j is highly correlated 
  with the PRF output F_sk, regardless of what the original LLM's probability was!

  score_i = Σ_{j=i+1}^{L} ln(1 / v_j)
  where v_j = F_sk            if x_j == 1
              1 - F_sk        if x_j == 0

  Detection threshold (Theorem 4):
    score_i > (L − i) + λ · √(L − i)

This makes detection incredibly fast, requiring only cryptographic hashing (HMAC)
and basic math, zero neural network inference required.
"""

import hmac
import hashlib
import math
import struct
from typing import Optional
from transformers import GPT2Tokenizer

from watermark import DEFAULT_KEY, DEFAULT_LAMBDA

def prf(key: bytes, seed_bits: list[int], position: int) -> float:
    """
    Pseudorandom function F_sk(r, pos) → [0, 1].
    Exactly matches the bit-level PRF from the generator.
    """
    byte_list = []
    for i in range(0, len(seed_bits), 8):
        chunk = seed_bits[i:i+8]
        byte_val = 0
        for bit in chunk:
            byte_val = (byte_val << 1) | bit
        if len(chunk) < 8:
            byte_val <<= (8 - len(chunk))
        byte_list.append(byte_val)
        
    seed_bytes = bytes(byte_list)
    pos_bytes = struct.pack(">I", position)
    
    digest = hmac.new(key, seed_bytes + pos_bytes, hashlib.sha256).digest()
    val = int.from_bytes(digest[:8], "big")
    return val / (2 ** 64)


class WatermarkDetector:
    """
    Detects bit-level watermarks planted by WatermarkGenerator.
    Iterates through candidate bit-prefixes, computing the correlation score.
    """

    def __init__(
        self,
        key: bytes = DEFAULT_KEY,
        lambda_: float = DEFAULT_LAMBDA,
        max_seed_bits: int = 800,   # Equivalent to checking the first ~50 tokens
        min_bits: int = 960,        # Minimum remaining bits to score (~60 tokens)
        model_name: str = "gpt2",
        tokenizer: Optional[GPT2Tokenizer] = None,
    ):
        self.key = key
        self.lambda_ = lambda_
        self.max_seed_bits = max_seed_bits
        self.min_bits = min_bits
        
        # We only need the tokenizer to convert the raw text string into token IDs.
        # No neural network (GPT2LMHeadModel) is loaded!
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            
        self.vocab_size = len(self.tokenizer)
        self.num_bits = math.ceil(math.log2(self.vocab_size))

    def _tokens_to_bits(self, tokens: list[int]) -> list[int]:
        """Convert a sequence of token IDs to their padded binary representations."""
        bits = []
        for token in tokens:
            # Handle out-of-bounds tokens gracefully
            t = min(token, self.vocab_size - 1)
            for j in range(self.num_bits):
                shift = self.num_bits - 1 - j
                bits.append((t >> shift) & 1)
        return bits

    def score_all_prefixes(self, bits: list[int]) -> list[dict]:
        """
        Compute detection scores for candidate seed bit-prefixes.
        (Implements Algorithm 4 verbatim).
        """
        L = len(bits)
        results = []
        
        # i represents the length of the candidate seed (r) in bits
        # We start from 1 up to max_seed_bits
        for i in range(1, min(L, self.max_seed_bits + 1)):
            remaining = L - i
            if remaining < self.min_bits:
                continue
                
            seed_bits = bits[:i]
            score_sum = 0.0
            
            # Score all subsequent bits
            for t_rel in range(remaining):
                j = i + t_rel  # global bit position
                
                u = prf(self.key, seed_bits, j)
                x_j = bits[j]
                
                # Equation from Algorithm 4:
                # v_j = x_j * F_sk + (1 - x_j) * (1 - F_sk)
                v = u if x_j == 1 else 1.0 - u
                
                # Clamp to avoid math domain errors (log(0))
                v = max(min(v, 1.0 - 1e-10), 1e-10)
                
                score_sum += math.log(1.0 / v)
                
            # Detection threshold from Theorem 4: (L - i) + \lambda * sqrt(L - i)
            threshold = remaining + self.lambda_ * math.sqrt(remaining)
            
            results.append({
                "seed_bit_pos": i,
                "score": score_sum,
                "length_bits": remaining,
                "threshold": threshold,
                "detected": score_sum > threshold,
            })
            
        return results

    def score(self, text: str) -> tuple[float, bool, int]:
        """
        Returns: (best_score, is_watermarked, best_seed_bit_position)
        """
        tokens = self.tokenizer.encode(text)
        bits = self._tokens_to_bits(tokens)

        if len(bits) < self.min_bits + 10:
            return 0.0, False, -1

        results = self.score_all_prefixes(bits)

        if not results:
            return 0.0, False, -1

        # Find the candidate seed that exceeds the threshold by the largest margin
        best = max(results, key=lambda r: r["score"] - r["threshold"])
        is_watermarked = best["score"] > best["threshold"]
        
        return best["score"], is_watermarked, best["seed_bit_pos"]

    def detect(self, text: str) -> bool:
        """Return True if the text is detected as watermarked."""
        _, detected, _ = self.score(text)
        return detected


if __name__ == "__main__":
    from watermark import WatermarkGenerator

    key = b"spc_feh_2026_watermark_key"
    prompt = "Artificial intelligence and machine learning are transforming"

    print("=== Watermark Detection Demo ===\n")

    # Generate watermarked text (Requires GPU/CPU time to run LLM)
    gen = WatermarkGenerator(key=key)
    result = gen.generate(prompt, max_new_tokens=60)
    wm_text = result["text"]
    print(f"Watermarked text:\n{wm_text}\n")

    # Detect watermark (Blazing fast, NO LLM REQUIRED!)
    det = WatermarkDetector(key=key)

    score, detected, pos = det.score(wm_text)
    print(f"[Watermarked Text] score={score:.2f}, detected={detected}, seed_bit_pos={pos}")

    # Try detection with wrong key
    det_wrong = WatermarkDetector(key=b"wrong_key_xyz")
    score_w, detected_w, _ = det_wrong.score(wm_text)
    print(f"[Wrong Key]        score={score_w:.2f}, detected={detected_w}")