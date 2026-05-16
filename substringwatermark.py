"""
substringwatermark.py - Substring-complete LLM watermark generation.

This implements the rolling-seed idea from Section 4.4 on top of the same
binary alphabet reduction used by watermark.py.  The practical difference from
the base generator is that the PRF position is local to the active seed block
and resets whenever a new seed block is adopted.  That makes a later substring
self-contained: if it contains one seed block and enough following bits, the
detector can verify it without knowing the original document prefix.
"""

import math
from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from watermark import DEFAULT_KEY, DEFAULT_LAMBDA, prf


def seed_update_threshold(lambda_: float, length_bits: int) -> float:
    """Entropy threshold for adopting the current block as the next seed."""
    if length_bits <= 0:
        return math.inf
    return (2.0 / math.log(2.0)) * lambda_ * math.sqrt(length_bits)


class SubstringWatermarkGenerator:
    """
    Generate watermarked text with periodically refreshed seed blocks.

    Seed blocks are aligned to token boundaries in this implementation.  The
    paper's algorithms are alphabet-generic; because this project reduces GPT-2
    tokens to 16-bit symbols, token alignment keeps the Streamlit demo and the
    substring detector tractable while preserving the rolling-seed mechanism.
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

        self.vocab_size = len(self.tokenizer)
        self.num_bits = math.ceil(math.log2(self.vocab_size))

        tokens = torch.arange(self.vocab_size)
        self.bit_matrix = torch.zeros((self.vocab_size, self.num_bits), dtype=torch.bool)
        for j in range(self.num_bits):
            shift = self.num_bits - 1 - j
            self.bit_matrix[:, j] = ((tokens >> shift) & 1).bool()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200) -> dict:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        device = self.model.device
        self.bit_matrix = self.bit_matrix.to(device)

        generated_bits: list[int] = []
        generated_tokens: list[int] = []
        entropy_history: list[float] = []
        rolling_entropy_history: list[float] = []
        prf_values: list[float] = []
        seed_events: list[dict] = []

        seed_bits: Optional[list[int]] = None
        initial_entropy = 0.0
        active_seed_start = 0
        local_prf_position = 0

        block_bits: list[int] = []
        block_entropy = 0.0
        block_start_bit = 0

        outputs = self.model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

        for _ in range(max_new_tokens):
            probs = probs.squeeze()
            valid_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
            current_pool_prob = 1.0
            token_bits = []

            for j in range(self.num_bits):
                bit_1_mask = valid_mask & self.bit_matrix[:, j]
                prob_1_unnorm = probs[bit_1_mask].sum().item()
                p_1 = prob_1_unnorm / current_pool_prob if current_pool_prob > 1e-12 else 0.0
                p_1 = max(0.0, min(1.0, p_1))
                p_0 = 1.0 - p_1

                if seed_bits is None:
                    bit = 1 if torch.rand(1).item() <= p_1 else 0
                    bit_prob = p_1 if bit == 1 else p_0
                    initial_entropy += -math.log(bit_prob + 1e-15)
                    entropy_history.append(initial_entropy)
                else:
                    if p_1 <= 1e-9:
                        bit = 0
                    elif p_1 >= 1.0 - 1e-9:
                        bit = 1
                    else:
                        u = prf(self.key, seed_bits, local_prf_position)
                        prf_values.append(u)
                        bit = 1 if u <= p_1 else 0
                    bit_prob = p_1 if bit == 1 else p_0
                    block_bits.append(bit)
                    block_entropy += -math.log(bit_prob + 1e-15)
                    rolling_entropy_history.append(block_entropy)
                    local_prf_position += 1

                generated_bits.append(bit)
                token_bits.append(bit)

                if bit == 1:
                    valid_mask = bit_1_mask
                    current_pool_prob = prob_1_unnorm
                else:
                    valid_mask = valid_mask & (~self.bit_matrix[:, j])
                    current_pool_prob = current_pool_prob - prob_1_unnorm

            token_id = int("".join(str(b) for b in token_bits), 2)
            if token_id >= self.vocab_size:
                token_id = self.tokenizer.eos_token_id

            generated_tokens.append(token_id)

            # Token-boundary initial seed lock.  This keeps later seed blocks
            # token-aligned, which is important for substring detection on text.
            if seed_bits is None and initial_entropy >= self.lambda_:
                seed_bits = list(generated_bits)
                active_seed_start = 0
                block_start_bit = len(generated_bits)
                local_prf_position = 0
                seed_events.append({
                    "kind": "initial",
                    "seed_start_bit": active_seed_start,
                    "seed_bit_length": len(seed_bits),
                    "seed_start_token": active_seed_start // self.num_bits,
                    "seed_token_length": len(seed_bits) // self.num_bits,
                    "entropy": initial_entropy,
                    "threshold": self.lambda_,
                })

            elif seed_bits is not None and block_bits:
                threshold = seed_update_threshold(self.lambda_, len(block_bits))
                if block_entropy >= threshold:
                    seed_bits = list(block_bits)
                    active_seed_start = block_start_bit
                    seed_events.append({
                        "kind": "rolling",
                        "seed_start_bit": active_seed_start,
                        "seed_bit_length": len(seed_bits),
                        "seed_start_token": active_seed_start // self.num_bits,
                        "seed_token_length": len(seed_bits) // self.num_bits,
                        "entropy": block_entropy,
                        "threshold": threshold,
                    })
                    block_bits = []
                    block_entropy = 0.0
                    block_start_bit = len(generated_bits)
                    local_prf_position = 0

            if token_id == self.tokenizer.eos_token_id:
                break

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
            "seed_bit_length": seed_events[0]["seed_bit_length"] if seed_events else None,
            "entropy_reached": initial_entropy,
            "entropy_history": entropy_history,
            "rolling_entropy_history": rolling_entropy_history,
            "prf_values": prf_values,
            "seed_events": seed_events,
        }


if __name__ == "__main__":
    gen = SubstringWatermarkGenerator()
    result = gen.generate("The study of cryptography has a long history", max_new_tokens=120)
    print(result["text"])
    print(f"Seed events: {len(result['seed_events'])}")
    for event in result["seed_events"]:
        print(event)
