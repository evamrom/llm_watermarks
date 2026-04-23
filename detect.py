"""
detect.py — Watermark Detector (Christ, Gunn, Zamir 2023)

Implements the detection algorithm from Section 4.3 (Algorithm 4 / Theorem 4).

Key idea:
  Given text and secret key, try every prefix position i as a candidate seed r.
  For each candidate, compute a score measuring how well the PRF values align
  with the actual token choices via the inverse-CDF mapping:

    score_i = Σ_{t=i}^{L-1}  log(1 / v_t)

  where  v_t = u_t        if inverse_cdf(probs, u_t) == actual_token_t   (match)
              1 - u_t     otherwise                                        (no match)

  Watermarked text: u_t is ALWAYS correlated with tokens[t] → score is anomalously high.
  Human text:      u_t is independent of tokens[t] → score ≈ (L − i)  (baseline).

  Detection threshold (Theorem 4):
    score_i > (L − i) + λ · √(L − i)

  where λ is the entropy parameter used during watermarking.

Performance note:
  A single GPT-2 forward pass over the full token sequence returns logits for
  ALL positions simultaneously (O(1) passes), which we exploit here.
"""

import hmac
import hashlib
import math
import struct
from typing import Optional
import torch

from watermark import DEFAULT_KEY, DEFAULT_LAMBDA
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class WatermarkDetector:
    """
    Detects watermarks planted by WatermarkGenerator.

    The detector scans every prefix position as a candidate seed and returns
    True if any prefix yields an anomalously high alignment score.
    """

    def __init__(
        self,
        key: bytes = DEFAULT_KEY,
        lambda_: float = DEFAULT_LAMBDA,
        null_mean_per_token: float = 1.0,  # theoretical E[score/token] under H0 for any text
        max_seed_pos: int = 50,           # only try seed positions 0..max_seed_pos (seeds lock early)
        min_tokens: int = 60,             # minimum tokens after seed to score; filters out short human texts
        model_name: str = "gpt2",
        model: Optional[GPT2LMHeadModel] = None,
        tokenizer: Optional[GPT2Tokenizer] = None,
    ):
        self.key = key
        self.lambda_ = lambda_
        self.null_mean_per_token = null_mean_per_token
        self.max_seed_pos = max_seed_pos
        self.min_tokens = min_tokens
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            print(f"Loading {model_name} for detection...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.eval()

    @torch.no_grad()
    def _get_all_probs(self, tokens: list[int]) -> torch.Tensor:
        """
        Single forward pass → probability distributions for all positions.

        Returns:  all_probs[t] = softmax distribution for predicting token at position t+1
                  Shape: [L-1, vocab_size]  (no distribution for position 0)
        """
        input_tensor = torch.tensor([tokens])
        outputs = self.model(input_tensor)
        # logits[0, t, :] predicts token at position t+1 given tokens[0..t]
        all_probs = torch.softmax(outputs.logits[0, :-1, :], dim=-1)
        return all_probs  # shape [L-1, vocab_size]

    def score_all_prefixes(self, tokens: list[int]) -> list[dict]:
        """
        Compute detection scores for every candidate seed prefix position.

        Returns a list of dicts (one per prefix position i) with:
          - 'seed_pos':   i
          - 'score':      alignment score Σ log(1/v_t)
          - 'length':     number of tokens scored (L − i − 1)
          - 'threshold':  detection threshold for this i
          - 'detected':   bool, score > threshold
        """
        L = len(tokens)
        all_probs = self._get_all_probs(tokens)  # [L-1, vocab_size]

        # ── Optimisation: sort each position's vocab ONCE, reuse for all seeds ─
        # Without this, inverse_cdf_sample would sort 50K tokens for every
        # (seed_pos × token) pair — O(max_seed_pos × L × V log V).
        # Precomputing reduces it to O(L × V log V).
        sorted_indices_all = []   # sorted_indices_all[k]: sorted token ids at position k+1
        cumsum_all = []           # cumsum_all[k]:         CDF at position k+1
        for t_abs in range(1, L):
            sp, si = torch.sort(all_probs[t_abs - 1], descending=True)
            sorted_indices_all.append(si)
            cumsum_all.append(torch.cumsum(sp.double(), dim=0))

        # ── Precompute per-token bytes to speed up seed serialisation ──────────
        tok_bytes = [struct.pack(">I", tok) for tok in tokens]

        results = []
        for i in range(min(L - 1, self.max_seed_pos)):
            remaining = L - i - 1
            if remaining < self.min_tokens:
                continue

            seed_bytes = b"".join(tok_bytes[: i + 1])
            score_sum = 0.0

            for t_rel in range(remaining):
                t_abs = i + 1 + t_rel

                # PRF: HMAC-SHA256(key, seed_bytes || position_bytes)
                pos_bytes = struct.pack(">I", t_abs)
                digest = hmac.new(self.key, seed_bytes + pos_bytes, hashlib.sha256).digest()
                u = int.from_bytes(digest[:8], "big") / (2 ** 64)
                u_c = max(min(u, 1.0 - 1e-9), 1e-9)

                # Lookup token via precomputed CDF (no re-sort)
                si = sorted_indices_all[t_abs - 1]
                cs = cumsum_all[t_abs - 1]
                idx = min(int(torch.searchsorted(cs, torch.tensor(u_c, dtype=torch.float64))),
                          len(si) - 1)
                x_predicted = int(si[idx])

                x_t = tokens[t_abs]
                v = max(u if x_predicted == x_t else 1.0 - u, 1e-10)
                score_sum += math.log(1.0 / v)

            threshold = self.null_mean_per_token * remaining + self.lambda_ * math.sqrt(remaining)
            results.append({
                "seed_pos": i,
                "score": score_sum,
                "length": remaining,
                "threshold": threshold,
                "detected": score_sum > threshold,
            })

        return results

    def detect(self, text: str) -> bool:
        """Return True if the text is detected as watermarked."""
        _, detected, _ = self.score(text)
        return detected

    def score(self, text: str) -> tuple[float, bool, int]:
        """
        Compute best detection score across all prefix positions.

        Returns: (best_score, is_watermarked, best_seed_position)
        """
        tokens = self.tokenizer.encode(text)

        if len(tokens) < self.min_tokens + 2:
            return 0.0, False, -1

        results = self.score_all_prefixes(tokens)

        if not results:
            return 0.0, False, -1

        best = max(results, key=lambda r: r["score"] - r["threshold"])
        is_watermarked = best["score"] > best["threshold"]
        return best["score"], is_watermarked, best["seed_pos"]


if __name__ == "__main__":
    from watermark import WatermarkGenerator

    key = b"spc_feh_2026_watermark_key"
    prompt = "Artificial intelligence and machine learning are transforming"

    print("=== Watermark Detection Demo ===\n")

    # Generate watermarked text
    gen = WatermarkGenerator(key=key)
    result = gen.generate(prompt, max_new_tokens=80)
    wm_text = result["text"]
    print(f"Watermarked text:\n{wm_text}\n")

    # Detect watermark
    det = WatermarkDetector(key=key)

    score, detected, pos = det.score(wm_text)
    print(f"[Watermarked] score={score:.2f}, detected={detected}, seed_pos={pos}")

    # Try detection with wrong key
    det_wrong = WatermarkDetector(key=b"wrong_key_xyz")
    score_w, detected_w, _ = det_wrong.score(wm_text)
    print(f"[Wrong key]   score={score_w:.2f}, detected={detected_w}")
