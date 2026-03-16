# Undetectable LLM Watermarking

Implementation of **"Undetectable Watermarks for Language Models"**
by Christ, Gunn & Zamir (2023) — built on GPT-2 via HuggingFace.

> Course project for **Secure and Private Computing – Foundations of Ethical Hacking**
> University of St. Gallen (HSG), Institute of Computer Science, Spring 2026

---

## What is this?

Traditional watermarking schemes embed a detectable signal in LLM outputs by **biasing the output distribution** — making some tokens more likely than others. This is detectable even without the secret key.

This paper asks a harder question:

> *Can we watermark LLM outputs without changing the output distribution **at all**?*

The answer is **yes**, using a cryptographic PRF (pseudorandom function). The key insight:

- **Without the key**: watermarked text is computationally indistinguishable from unwatermarked text.
- **With the key**: a detector can identify watermarked text with high confidence.

---

## How it works

### Watermarking (Algorithm 3, §4.3)

```
Input:  prompt, secret key sk, entropy threshold λ
Output: watermarked text

1. Generate tokens naturally from GPT-2.
   Track cumulative empirical entropy: H += -log2(p_chosen_token)

2. Once H ≥ λ  →  lock seed  r = all tokens so far

3. For every subsequent token:
   u = HMAC-SHA256(sk, r, position)   ← PRF value in [0, 1]
   token = inverse_cdf(softmax_probs, u)
                     ↑
           Sort vocab by prob (desc), build CDF 0→1, pick at position u.
           Since u looks uniform to anyone without sk,
           the output distribution is provably UNCHANGED.
```

### Detection (Theorem 4)

```
Input:  text, secret key sk

For each prefix position i  (trying every possible seed):
    seed_candidate = tokens[:i]
    score = Σ  log(1 / v_t)   for t = i+1 … L
            where  v_t = u_t       if PRF(seed, t) would select the actual token
                   v_t = 1 − u_t  otherwise

    Threshold (Theorem 4):  score > (L − i) + λ · √(L − i)

Watermarked text:  u_t always selects the right token  →  score anomalously HIGH
Human text:        u_t is independent of tokens        →  score ≈ baseline (L − i)
```

---

## Project structure

```
llm_watermarjs/
│
├── watermark.py       # WatermarkGenerator — phases 1 & 2, PRF, inverse-CDF
├── detect.py          # WatermarkDetector  — prefix scan, scoring, threshold
├── experiments.py     # Soundness & Completeness tests + score distribution plot
├── requirements.txt   # Python dependencies
└── docs/
    ├── 2023-763.pdf               # Original paper
    ├── Implementation Concept.pdf # Pre-implementation design notes
    └── SPC-FEH_2026_Intro.pdf     # Course introduction slides
```

---

## Setup

**Requires Python 3.11 or 3.12** (PyTorch does not support Python 3.13 yet)

```bash
# Create virtual environment — use python3.12 explicitly
python3.12 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

GPT-2 weights (~500 MB) are downloaded automatically from HuggingFace on first run.

---

## Running the experiments

### Full experiment suite

```bash
python experiments.py
```

This runs two tests and produces `score_distribution.png`:

| Test | What it does | Expected result |
|------|-------------|-----------------|
| **Soundness** | Runs detector on 8 human-written texts | False positive rate ≈ 0% |
| **Completeness** | Generates 8 watermarked texts, runs detector | Detection rate ≈ 100% |

Sample output:
```
SOUNDNESS TEST — Human-Written Text
  [01] score=  82.14  detected=False
  [02] score=  79.33  detected=False
  ...
  False positive rate: 0.0%

COMPLETENESS TEST — Watermarked Text
  [01] Generating...  prompt='The history of cryptography begins in ancient ...'
        score= 148.62  detected=True  entropy=4.1bits  seed_at=12
  ...
  True detection rate: 100.0%

SUMMARY
  Human scores:       mean=81.20, std=6.43
  Watermarked scores: mean=147.91, std=11.05
  Score gap:          66.71
  False positive rate: 0.0%
  True detection rate: 100.0%
```

### Quick demos

```bash
# Demo watermarking only (generates one watermarked text)
python watermark.py

# Demo detection only (generates + detects + tests wrong key)
python detect.py
```

---

## Output: `score_distribution.png`

The experiment produces a two-panel figure:

- **Left**: histogram of detection scores — watermarked (red) vs. human (blue)
- **Right**: box plot comparing the two distributions

A successful implementation shows **clearly separated** distributions, confirming:
- Soundness: human text scores cluster at the baseline
- Completeness: watermarked text scores are far above the threshold

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key` | `b"spc_feh_2026_watermark_key"` | Secret key `sk` — must match between generator and detector |
| `lambda_` | `4.0` bits | Entropy threshold for seed locking; also controls detection threshold |
| `max_new_tokens` | `100–200` | More tokens → stronger detection signal |
| `min_tokens` | `10` | Minimum tokens after seed to attempt detection |

**Increasing `lambda_`** makes the seed lock later (more natural tokens first), which improves undetectability but requires more generated text for reliable detection.

---

## Design choices vs. the paper

| Paper | This implementation |
|-------|-------------------|
| Binary alphabet (§4.1 reduction, each token → ~17 bits) | Inverse-CDF directly on full ~50K vocab (practical simplification) |
| Random oracle `O` | HMAC-SHA256 PRF (standard replacement, §3.3) |
| Formal soundness: `negl(λ)` false-positive rate | Empirical soundness: threshold `(L−i) + λ√(L−i)` |
| Arbitrary LLM | GPT-2 (exposes full probability distributions, free) |

The full binary reduction (§4.1) would give stronger formal guarantees but is significantly more complex to implement. The inverse-CDF approach on the full vocabulary is the natural practical equivalent and preserves the core undetectability argument.

---

## Paper reference

> Miranda Christ, Sam Gunn, Or Zamir.
> **"Undetectable Watermarks for Language Models."**
> *STOC 2024 / arXiv:2306.09194*, May 2023.
> Columbia University · UC Berkeley · Princeton University
