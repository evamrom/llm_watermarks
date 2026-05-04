"""
watermark.py — Undetectable LLM Watermarking (Christ, Gunn, Zamir 2023)

Implements the exact binary alphabet reduction from Section 4.1 and the 
watermarking algorithm from Section 4.3 (Algorithm 3) of the paper.

Key changes from the Inverse-CDF version:
  - Vocabulary Reduction: The entire vocabulary is mapped to a binary alphabet. 
    A GPT-2 token (out of 50257) is represented as a 16-bit binary string.
  - Bit-by-Bit Sampling: Instead of sampling a whole token, the model evaluates 
    the probability of the next bit being '1' given the previous bits.
  - Entropy & PRF: Entropy is accumulated bit-by-bit. The PRF operates on the 
    bit-sequence seed (r) and outputs a bit choice directly based on p_i(1).
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
DEFAULT_LAMBDA = 4.0          # λ: entropy threshold for seed locking
# ──────────────────────────────────────────────────────────────────────────────


def prf(key: bytes, seed_bits: list[int], position: int) -> float:
    """
    Pseudorandom function F_sk(r, pos) → [0, 1].
    
    Operates on a binary sequence. Packs the seed bits into a tightly encoded 
    byte array to act as the HMAC message, appended with the position.
    """
    # Pack seed bits into bytes (left-aligned)
    byte_list = []
    for i in range(0, len(seed_bits), 8):
        chunk = seed_bits[i:i+8]
        byte_val = 0
        for bit in chunk:
            byte_val = (byte_val << 1) | bit
        if len(chunk) < 8:
            byte_val <<= (8 - len(chunk)) # Pad remaining bits with 0
        byte_list.append(byte_val)
        
    seed_bytes = bytes(byte_list)
    pos_bytes = struct.pack(">I", position)
    
    digest = hmac.new(key, seed_bytes + pos_bytes, hashlib.sha256).digest()
    val = int.from_bytes(digest[:8], "big")
    return val / (2 ** 64)


class WatermarkGenerator:
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

        # Precompute the binary representation matrix for the entire vocabulary
        self.vocab_size = len(self.tokenizer)
        # For GPT-2 (50257), we need 16 bits (2^15 = 32768, 2^16 = 65536)
        self.num_bits = math.ceil(math.log2(self.vocab_size))
        
        tokens = torch.arange(self.vocab_size)
        self.bit_matrix = torch.zeros((self.vocab_size, self.num_bits), dtype=torch.bool)
        for j in range(self.num_bits):
            shift = self.num_bits - 1 - j
            self.bit_matrix[:, j] = ((tokens >> shift) & 1).bool()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> dict:
        """
        Generate watermarked text using the exact Bit-by-Bit reduction.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        device = self.model.device
        self.bit_matrix = self.bit_matrix.to(device)

        global_bit_index = 0
        cumulative_entropy = 0.0
        
        seed_bits: Optional[list[int]] = None
        generated_bits: list[int] = []
        generated_tokens: list[int] = []

        outputs = self.model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

        for step in range(max_new_tokens):
            probs = probs.squeeze()
            
            # Mask to keep track of which tokens are still valid for the current bit prefix
            valid_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
            current_pool_prob = 1.0
            token_bits = []

            # ── Bit-by-Bit Generation Loop (Section 4.1) ──────────
            for j in range(self.num_bits):
                # Calculate probability of the next bit being 1: p'_{i,j}(1)
                bit_1_mask = valid_mask & self.bit_matrix[:, j]
                prob_1_unnorm = probs[bit_1_mask].sum().item()
                
                if current_pool_prob > 1e-12:
                    p_1 = prob_1_unnorm / current_pool_prob
                else:
                    p_1 = 0.0
                
                # Clamp for floating point safety
                p_1 = max(0.0, min(1.0, p_1))
                p_0 = 1.0 - p_1

                if seed_bits is None:
                    # Phase 1: Natural bit sampling
                    bit = 1 if torch.rand(1).item() <= p_1 else 0
                    
                    # Track bit-level entropy
                    bit_prob = p_1 if bit == 1 else p_0
                    cumulative_entropy += -math.log(bit_prob + 1e-15)

                    if cumulative_entropy >= self.lambda_:
                        # Lock the seed bits precisely at the bit where entropy crosses λ
                        seed_bits = list(generated_bits) + [bit]
                else:
                    # Phase 2: Watermark embedding using PRF
                    if p_1 <= 1e-9:
                        bit = 0
                    elif p_1 >= 1.0 - 1e-9:
                        bit = 1
                    else:
                        u = prf(self.key, seed_bits, global_bit_index)
                        bit = 1 if u <= p_1 else 0

                # Register bit
                generated_bits.append(bit)
                token_bits.append(bit)
                global_bit_index += 1

                # Update the valid pool for the next bit
                if bit == 1:
                    valid_mask = bit_1_mask
                    current_pool_prob = prob_1_unnorm
                else:
                    valid_mask = valid_mask & (~self.bit_matrix[:, j])
                    current_pool_prob = current_pool_prob - prob_1_unnorm

            # Reconstruct the token from the 16 generated bits
            token_id = int("".join(str(b) for b in token_bits), 2)
            
            # Guardrail: Out of bounds (e.g. bits resolved to > 50257 due to float drift)
            if token_id >= self.vocab_size:
                token_id = self.tokenizer.eos_token_id

            generated_tokens.append(token_id)

            if token_id == self.tokenizer.eos_token_id:
                break

            # ── Forward Pass for Next Token ──────────────────────────
            outputs = self.model(
                torch.tensor([[token_id]], device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "text": text,
            "generated_tokens": generated_tokens,
            "generated_bits": generated_bits,
            "seed_bits": seed_bits,
            "seed_bit_length": len(seed_bits) if seed_bits else None,
            "entropy_reached": cumulative_entropy,
        }


if __name__ == "__main__":
    gen = WatermarkGenerator()
    prompt = "The study of cryptography has a long history"
    print(f"Prompt: {prompt}\n")

    result = gen.generate(prompt, max_new_tokens=50)
    print(f"Generated text:\n{result['text']}\n")
    print(f"Entropy at seed lock: {result['entropy_reached']:.2f} nats")
    print(f"Seed locked at bit position: {result['seed_bit_length']}")
    print(f"Total bits generated: {len(result['generated_bits'])}")