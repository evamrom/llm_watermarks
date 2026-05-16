"""
substringdetect.py - Detector for substring-complete rolling-seed watermarks.

The detector searches for a candidate seed block inside the submitted text and
then checks whether the following bits match the PRF stream generated from that
seed.  PRF positions are relative to the candidate seed, not absolute document
positions, so a truncated middle segment can still be detected.
"""

import math
import hmac
import hashlib
import struct
from typing import Optional

try:
    from transformers import GPT2Tokenizer
except ModuleNotFoundError:
    GPT2Tokenizer = None

DEFAULT_KEY = b"spc_feh_2026_watermark_key"
DEFAULT_LAMBDA = 4.0


def prf(key: bytes, seed_bits: list[int], position: int) -> float:
    """Pseudorandom function F_sk(r, pos) -> [0, 1]."""
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


class SubstringWatermarkDetector:
    """Detect rolling-seed substring watermarks."""

    def __init__(
        self,
        key: bytes = DEFAULT_KEY,
        lambda_: float = DEFAULT_LAMBDA,
        min_seed_bits: int = 64,
        max_seed_bits: int = 800,
        min_score_bits: int = 512,
        max_score_bits: int = 1536,
        start_step_bits: Optional[int] = None,
        seed_step_bits: Optional[int] = None,
        score_step_bits: Optional[int] = None,
        model_name: str = "gpt2",
        tokenizer: Optional["GPT2Tokenizer"] = None,
    ):
        self.key = key
        self.lambda_ = lambda_
        self.min_seed_bits = min_seed_bits
        self.max_seed_bits = max_seed_bits
        self.min_score_bits = min_score_bits
        self.max_score_bits = max_score_bits

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if GPT2Tokenizer is None:
                raise ModuleNotFoundError(
                    "transformers is required when no tokenizer is provided"
                )
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        self.vocab_size = len(self.tokenizer)
        self.num_bits = math.ceil(math.log2(self.vocab_size))

        # Defaults match the token-aligned seed blocks emitted by
        # SubstringWatermarkGenerator.
        self.start_step_bits = start_step_bits or self.num_bits
        self.seed_step_bits = seed_step_bits or self.num_bits
        self.score_step_bits = score_step_bits or (4 * self.num_bits)

    def _tokens_to_bits(self, tokens: list[int]) -> list[int]:
        bits = []
        for token in tokens:
            t = min(token, self.vocab_size - 1)
            for j in range(self.num_bits):
                shift = self.num_bits - 1 - j
                bits.append((t >> shift) & 1)
        return bits

    def _threshold(self, length_bits: int) -> float:
        return length_bits + self.lambda_ * math.sqrt(length_bits)

    def _empty(self) -> dict:
        return {
            "score": 0.0,
            "detected": False,
            "seed_bit_pos": -1,
            "seed_bit_length": 0,
            "score_bit_length": 0,
            "threshold": 0.0,
            "all_results": [],
            "trajectory": [],
            "traj_thresholds": [],
        }

    def score_all_substrings(self, bits: list[int]) -> list[dict]:
        """
        Search token-aligned candidate seed windows and score their suffixes.

        For each seed candidate we compute one cumulative trajectory up to
        max_score_bits and evaluate it at score_step_bits checkpoints.  This
        keeps the search practical while still trying many possible substring
        starts, seed lengths, and verification lengths.
        """
        L = len(bits)
        results = []
        max_start = L - self.min_seed_bits - self.min_score_bits
        if max_start < 0:
            return results

        for seed_start in range(0, max_start + 1, self.start_step_bits):
            max_seed_here = min(self.max_seed_bits, L - seed_start - self.min_score_bits)
            for seed_len in range(self.min_seed_bits, max_seed_here + 1, self.seed_step_bits):
                score_start = seed_start + seed_len
                available = L - score_start
                if available < self.min_score_bits:
                    continue

                score_limit = min(self.max_score_bits, available)
                seed_bits = bits[seed_start:score_start]
                running = 0.0

                for t_rel in range(score_limit):
                    x_j = bits[score_start + t_rel]
                    u = prf(self.key, seed_bits, t_rel)
                    v = u if x_j == 1 else 1.0 - u
                    v = max(min(v, 1.0 - 1e-10), 1e-10)
                    running += math.log(1.0 / v)

                    n = t_rel + 1
                    if n < self.min_score_bits or n % self.score_step_bits != 0:
                        continue

                    threshold = self._threshold(n)
                    results.append({
                        "seed_bit_pos": seed_start,
                        "seed_bit_length": seed_len,
                        "score_bit_length": n,
                        "score": running,
                        "threshold": threshold,
                        "detected": running > threshold,
                    })

        return results

    def score_bits(self, bits: list[int]) -> tuple[float, bool, int, int, int]:
        all_results = self.score_all_substrings(bits)
        if not all_results:
            return 0.0, False, -1, 0, 0

        best = max(all_results, key=lambda r: r["score"] - r["threshold"])
        return (
            best["score"],
            best["score"] > best["threshold"],
            best["seed_bit_pos"],
            best["seed_bit_length"],
            best["score_bit_length"],
        )

    def score_tokens(self, tokens: list[int]) -> tuple[float, bool, int, int, int]:
        return self.score_bits(self._tokens_to_bits(tokens))

    def score(self, text: str) -> tuple[float, bool, int, int, int]:
        tokens = self.tokenizer.encode(text)
        return self.score_tokens(tokens)

    def detect(self, text: str) -> bool:
        _, detected, _, _, _ = self.score(text)
        return detected

    def score_details(self, text: Optional[str] = None, bits: Optional[list[int]] = None) -> dict:
        if bits is None:
            if text is None:
                return self._empty()
            tokens = self.tokenizer.encode(text)
            bits = self._tokens_to_bits(tokens)

        all_results = self.score_all_substrings(bits)
        if not all_results:
            return self._empty()

        best = max(all_results, key=lambda r: r["score"] - r["threshold"])
        detected = best["score"] > best["threshold"]

        seed_start = best["seed_bit_pos"]
        seed_end = seed_start + best["seed_bit_length"]
        score_len = best["score_bit_length"]
        seed_bits = bits[seed_start:seed_end]

        trajectory, traj_thresholds = [], []
        running = 0.0
        for t_rel in range(score_len):
            x_j = bits[seed_end + t_rel]
            u = prf(self.key, seed_bits, t_rel)
            v = u if x_j == 1 else 1.0 - u
            v = max(min(v, 1.0 - 1e-10), 1e-10)
            running += math.log(1.0 / v)
            n = t_rel + 1
            trajectory.append(running)
            traj_thresholds.append(self._threshold(n))

        return {
            "score": best["score"],
            "detected": detected,
            "seed_bit_pos": seed_start,
            "seed_bit_length": best["seed_bit_length"],
            "score_bit_length": score_len,
            "threshold": best["threshold"],
            "all_results": all_results,
            "trajectory": trajectory,
            "traj_thresholds": traj_thresholds,
        }


if __name__ == "__main__":
    from substringwatermark import SubstringWatermarkGenerator

    key = b"spc_feh_2026_watermark_key"
    gen = SubstringWatermarkGenerator(key=key)
    result = gen.generate("Artificial intelligence and machine learning are transforming", 180)
    det = SubstringWatermarkDetector(key=key, tokenizer=gen.tokenizer)

    tokens = result["generated_tokens"]
    middle_tokens = tokens[len(tokens) // 3:]
    score, detected, pos, seed_len, score_len = det.score_tokens(middle_tokens)
    print(f"score={score:.2f}, detected={detected}, seed={pos}+{seed_len}, scored={score_len}")
