"""
experiments.py — Empirical Evaluation of the Watermarking Scheme

Evaluates two core properties from the paper (Section 2.4 / Theorem 3):

  1. Soundness  (Definition 6): Detector returns False on human-written text.
                                → measures false-positive rate

  2. Completeness (Definition 7): Detector returns True on watermarked text.
                                  → measures true-detection rate

  3. Score separation: plots score distributions for both cases.
     Correctly implemented watermarking should show clearly separated curves.

Usage:
    python experiments.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from watermark import WatermarkGenerator
from detect import WatermarkDetector


# ── Prompts for watermarked generation ────────────────────────────────────────
PROMPTS = [
    "The history of cryptography begins in ancient civilisations",
    "In quantum mechanics, the Heisenberg uncertainty principle states",
    "Climate change is driven by the accumulation of greenhouse gases",
    "The French Revolution fundamentally altered the political landscape",
    "Machine learning algorithms learn patterns from large datasets",
    "The discovery of DNA structure in 1953 transformed biology",
    "Language models are trained on vast amounts of text data",
    "The laws of thermodynamics govern energy transformations in systems",
]

# ── Human-written reference texts (not model-generated) ──────────────────────
HUMAN_TEXTS = [
    # Wikipedia-style excerpts
    (
        "Mathematics is an area of knowledge that includes the topics of numbers, "
        "formulas and related structures, shapes and the spaces in which they are "
        "contained, and quantities and their changes. Most mathematical activity "
        "involves the discovery of properties of abstract objects and the use of "
        "pure reason to prove them."
    ),
    (
        "The ocean covers more than seventy percent of Earth's surface. It is "
        "divided into five named oceans: the Pacific, Atlantic, Indian, Arctic, "
        "and Southern Oceans. Seawater contains roughly 3.5 percent dissolved "
        "salts on average, making it undrinkable for most organisms without treatment."
    ),
    (
        "Philosophy is the systematic study of general and fundamental questions "
        "about existence, knowledge, values, reason, mind, and language. It is "
        "distinguished from other ways of addressing such questions by its critical, "
        "generally systematic approach and its reliance on rational argument."
    ),
    (
        "A computer is a digital electronic machine that can be programmed to carry "
        "out sequences of arithmetic or logical operations automatically. Modern "
        "computers can perform generic sets of operations known as programs, which "
        "enable them to perform an extremely wide range of tasks."
    ),
    (
        "Music is the art of arranging sounds in time to produce a composition "
        "through the elements of melody, harmony, rhythm, and timbre. It is one "
        "of the universal cultural aspects of all human societies. General "
        "definitions of music include common elements such as pitch, rhythm, "
        "dynamics, and the sonic qualities of timbre and texture."
    ),
    (
        "The immune system is a network of biological processes that protects an "
        "organism from diseases. It detects and responds to a wide variety of "
        "pathogens, from viruses to parasitic worms, as well as cancer cells and "
        "objects such as wood splinters, distinguishing them from the organism's "
        "own healthy tissue."
    ),
    (
        "Architecture is the art and technique of designing and building, as "
        "distinguished from the skills associated with construction. It is both "
        "the process and the product of sketching, conceiving, planning, designing, "
        "and constructing buildings or other structures."
    ),
    (
        "The solar system consists of the Sun and everything gravitationally bound "
        "to it, including the eight planets and their moons, the asteroid belt, "
        "comets, and various smaller bodies. The four inner planets are rocky "
        "terrestrial worlds, while the outer four are giant planets."
    ),
]


def run_soundness_experiment(
    detector: WatermarkDetector,
    human_texts: list[str],
    verbose: bool = True,
) -> tuple[list[float], float]:
    """
    Soundness test: run detector on human-written text.
    Expected: false-positive rate ≈ 0 (or very small).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SOUNDNESS TEST — Human-Written Text")
        print("=" * 60)

    scores = []
    false_positives = 0

    for i, text in enumerate(human_texts):
        n_tokens = len(detector.tokenizer.encode(text))
        score, detected, pos = detector.score(text)
        # Approximate threshold display (actual threshold in detect.py uses remaining at best pos)
        threshold = detector.null_mean_per_token * n_tokens + detector.z_score * math.sqrt(n_tokens)
        scores.append(score)
        if detected:
            false_positives += 1
        if verbose:
            print(f"  [{i+1:02d}] score={score:7.2f}  threshold≈{threshold:6.1f}  detected={detected}")

    fpr = false_positives / len(human_texts)
    if verbose:
        print(f"\n  False positive rate: {fpr:.1%}  ({false_positives}/{len(human_texts)})")

    return scores, fpr


def run_completeness_experiment(
    generator: WatermarkGenerator,
    detector: WatermarkDetector,
    prompts: list[str],
    max_new_tokens: int = 100,
    verbose: bool = True,
) -> tuple[list[float], float]:
    """
    Completeness test: generate watermarked text and run detector.
    Expected: detection rate ≈ 100%.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("COMPLETENESS TEST — Watermarked Text")
        print("=" * 60)

    scores = []
    detections = 0

    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"  [{i+1:02d}] Generating...  prompt='{prompt[:45]}...'")

        result = generator.generate(prompt, max_new_tokens=max_new_tokens)
        wm_text = result["text"]
        seed_pos = result["seed_position"]
        entropy = result["entropy_reached"]

        score, detected, best_pos = detector.score(wm_text, prompt)
        n_tokens = len(detector.tokenizer.encode(wm_text))
        threshold = detector.null_mean_per_token * n_tokens + detector.z_score * math.sqrt(n_tokens)
        scores.append(score)
        if detected:
            detections += 1

        if verbose:
            print(
                f"        score={score:7.2f}  threshold≈{threshold:6.1f}  detected={detected}"
                f"  entropy={entropy:.1f}bits  seed_at={seed_pos}"
            )

    tdr = detections / len(prompts)
    if verbose:
        print(f"\n  True detection rate: {tdr:.1%}  ({detections}/{len(prompts)})")

    return scores, tdr


def plot_score_distributions(
    human_scores: list[float],
    wm_scores: list[float],
    lambda_: float,
    output_path: str = "score_distribution.png",
) -> None:
    """
    Plot score distributions for watermarked vs human text.
    Well-separated distributions confirm the watermarking scheme works.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Histogram ────────────────────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(
        min(human_scores + wm_scores) - 5,
        max(human_scores + wm_scores) + 5,
        25,
    )
    ax.hist(human_scores, bins=bins, alpha=0.7, label="Human text", color="#4477aa", density=True)
    ax.hist(wm_scores, bins=bins, alpha=0.7, label="Watermarked text", color="#cc4455", density=True)
    ax.set_xlabel("Detection Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distributions", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Box plot ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    bp = ax2.boxplot(
        [human_scores, wm_scores],
        labels=["Human", "Watermarked"],
        patch_artist=True,
        notch=False,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4477aa")
    bp["boxes"][1].set_facecolor("#cc4455")
    for patch in bp["boxes"]:
        patch.set_alpha(0.7)
    ax2.set_ylabel("Detection Score", fontsize=12)
    ax2.set_title("Score Comparison", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotate with stats
    info = (
        f"Human:      μ={np.mean(human_scores):.1f}, σ={np.std(human_scores):.1f}\n"
        f"Watermarked: μ={np.mean(wm_scores):.1f}, σ={np.std(wm_scores):.1f}"
    )
    fig.text(0.5, 0.01, info, ha="center", fontsize=10, style="italic")

    plt.suptitle(
        "Undetectable LLM Watermark — Christ, Gunn, Zamir 2023\n"
        f"GPT-2, λ={lambda_}, n_human={len(human_scores)}, n_watermarked={len(wm_scores)}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to '{output_path}'")


def run_all(max_new_tokens: int = 100) -> None:
    """Run the full experimental evaluation."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("\n" + "=" * 60)
    print("Initialising models (this may take a moment)...")
    print("=" * 60)

    # Load once, share between generator and detector
    print("Loading gpt2 (shared)...")
    shared_model = GPT2LMHeadModel.from_pretrained("gpt2")
    shared_model.eval()
    shared_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    generator = WatermarkGenerator(model=shared_model, tokenizer=shared_tokenizer)
    detector = WatermarkDetector(model=shared_model, tokenizer=shared_tokenizer)

    # ── Soundness ─────────────────────────────────────────────────────────────
    human_scores, fpr = run_soundness_experiment(detector, HUMAN_TEXTS)

    # ── Completeness ──────────────────────────────────────────────────────────
    wm_scores, tdr = run_completeness_experiment(
        generator, detector, PROMPTS, max_new_tokens=max_new_tokens
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_score_distributions(human_scores, wm_scores, lambda_=detector.lambda_)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Human scores:       mean={np.mean(human_scores):.2f}, std={np.std(human_scores):.2f}")
    print(f"  Watermarked scores: mean={np.mean(wm_scores):.2f}, std={np.std(wm_scores):.2f}")
    print(f"  Score gap:          {np.mean(wm_scores) - np.mean(human_scores):.2f}")
    print(f"  False positive rate: {fpr:.1%}")
    print(f"  True detection rate: {tdr:.1%}")

    # Sanity check: higher watermarked scores = scheme is working
    gap = np.mean(wm_scores) - np.mean(human_scores)
    if gap > 0:
        print(f"\n  Watermarked scores are {gap:.1f} points HIGHER on average — scheme working.")
    else:
        print(f"\n  WARNING: watermarked scores not higher — check lambda / token count.")


if __name__ == "__main__":
    run_all(max_new_tokens=200)
