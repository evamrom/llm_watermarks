"""
app.py — Streamlit UI for Undetectable LLM Watermarking
Christ, Gunn & Zamir (2023) — Course project SPC-FEH 2026, HSG
"""

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from watermark import WatermarkGenerator, DEFAULT_KEY, DEFAULT_LAMBDA
from detect import WatermarkDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Undetectable LLM Watermarking",
    layout="wide",
)

# ── Load model once (cached across reruns) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading GPT-2 (one-time, ~500 MB)...")
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("LLM Watermarking")
st.sidebar.markdown(
    "**Christ, Gunn & Zamir (2023)**\n\n"
    "Undetectable watermarks using a cryptographic PRF.\n\n"
    "---"
)
page = st.sidebar.radio(
    "Navigate",
    ["1 · Generate", "2 · Detect", "3 · Experiments"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Parameters**")
secret_key = st.sidebar.text_input("Secret key", value="spc_feh_2026_watermark_key")
lambda_ = st.sidebar.slider("λ (entropy threshold, bits)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
key_bytes = secret_key.encode()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
if page == "1 · Generate":
    st.title("Phase 1 & 2 — Watermarked Text Generation")
    st.markdown(
        "Enter a prompt and generate watermarked text. "
        "Tokens are colour-coded by phase: "
        "**:blue[blue = Phase 1 (natural sampling)]** · "
        "**:red[red = Phase 2 (PRF-guided, watermarked)]**"
    )

    prompt = st.text_input(
        "Prompt",
        value="The history of cryptography begins in ancient civilisations",
    )
    max_tokens = st.slider("Max new tokens", min_value=5, max_value=50, value=30, step=5)

    if st.button("Generate watermarked text", type="primary"):
        with st.spinner("Generating..."):
            gen = WatermarkGenerator(key=key_bytes, lambda_=lambda_, model=model, tokenizer=tokenizer)
            result = gen.generate(prompt, max_new_tokens=max_tokens)

        # ── Stats ─────────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("Entropy at seed lock", f"{result['entropy_reached']:.2f} bits")
        col2.metric("Seed locked at token", result["seed_position"] if result["seed_position"] else "never")
        col3.metric("Tokens generated", len(result["generated_tokens"]))

        st.markdown("---")

        # ── Colour-coded token display ─────────────────────────────────────────
        if result["seed_position"] is None:
            st.warning("Seed never locked — entropy threshold not reached. Try a lower λ or more tokens.")
            st.write(result["text"])
        else:
            prompt_len = len(tokenizer.encode(prompt))
            seed_pos = result["seed_position"]

            # Split generated tokens into phase 1 and phase 2
            phase1_tokens = result["all_tokens"][prompt_len:seed_pos]
            phase2_tokens = result["all_tokens"][seed_pos:]

            phase1_text = tokenizer.decode(phase1_tokens, skip_special_tokens=True)
            phase2_text = tokenizer.decode(phase2_tokens, skip_special_tokens=True)

            st.markdown("#### Generated text (colour-coded by phase)")
            st.markdown(
                f'<div style="font-size:1.1rem; line-height:1.8; padding:1rem; '
                f'background:#f8f9fa; border-radius:8px; border:1px solid #dee2e6;">'
                f'<span style="color:#1d6fa4; font-weight:500;">{phase1_text} </span>'
                f'<span style="color:#c0392b; font-weight:500;">{phase2_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            st.markdown(
                "**Blue** = Phase 1 (natural random sampling, building entropy)  \n"
                "**Red** = Phase 2 (PRF-guided via inverse-CDF — the watermark)"
            )

            with st.expander("Show full token list"):
                st.write(result["all_tokens"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DETECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2 · Detect":
    st.title("Detection — Does this text carry our watermark?")
    st.markdown(
        "Paste any text below. The detector will try every prefix position as a "
        "candidate seed and report the best alignment score."
    )

    text_input = st.text_area(
        "Text to analyse",
        height=200,
        placeholder="Paste text here — watermarked or human-written...",
    )

    col_a, col_b = st.columns(2)
    run_correct = col_a.button("Detect (correct key)", type="primary")
    run_wrong   = col_b.button("Detect (wrong key)", type="secondary")

    if run_correct or run_wrong:
        if not text_input.strip():
            st.error("Please enter some text to analyse.")
        else:
            used_key = key_bytes if run_correct else b"wrong_key_xyz"
            label = "correct key" if run_correct else "WRONG key"

            with st.spinner(f"Running detector with {label}..."):
                det = WatermarkDetector(key=used_key, lambda_=lambda_, model=model, tokenizer=tokenizer)
                score, detected, seed_pos = det.score(text_input)
                tokens = tokenizer.encode(text_input)
                n = len(tokens)
                threshold = det.null_mean_per_token * n + det.lambda_ * math.sqrt(n)

            st.markdown("---")

            # ── Result banner ──────────────────────────────────────────────────
            if detected:
                st.success(f"WATERMARK DETECTED  (key: {label})")
            else:
                st.error(f"No watermark detected  (key: {label})")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Score", f"{score:.2f}")
            col2.metric("Threshold", f"{threshold:.2f}")
            col3.metric("Gap", f"{score - threshold:+.2f}")
            col4.metric("Best seed position", seed_pos if seed_pos >= 0 else "n/a")

            # ── Score bar ──────────────────────────────────────────────────────
            st.markdown("#### Score vs Threshold")
            fig, ax = plt.subplots(figsize=(8, 1.5))
            max_val = max(score, threshold) * 1.2
            ax.barh(["Score"], [score], color="#c0392b" if detected else "#4477aa", height=0.4)
            ax.axvline(threshold, color="black", linewidth=2, linestyle="--", label=f"Threshold ({threshold:.1f})")
            ax.set_xlim(0, max_val)
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlabel("Score")
            ax.spines[["top", "right", "left"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "3 · Experiments":
    st.title("Experiments — Soundness & Completeness")
    st.markdown(
        "Run the full empirical evaluation from the paper:\n\n"
        "- **Soundness**: detector on human-written text → should give 0 false positives\n"
        "- **Completeness**: detector on watermarked text → should detect 100%"
    )

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

    HUMAN_TEXTS = [
        "Mathematics is an area of knowledge that includes the topics of numbers, formulas and related structures, shapes and the spaces in which they are contained, and quantities and their changes.",
        "The ocean covers more than seventy percent of Earth's surface. It is divided into five named oceans: the Pacific, Atlantic, Indian, Arctic, and Southern Oceans.",
        "Philosophy is the systematic study of general and fundamental questions about existence, knowledge, values, reason, mind, and language.",
        "A computer is a digital electronic machine that can be programmed to carry out sequences of arithmetic or logical operations automatically.",
        "Music is the art of arranging sounds in time to produce a composition through the elements of melody, harmony, rhythm, and timbre.",
        "The immune system is a network of biological processes that protects an organism from diseases. It detects and responds to a wide variety of pathogens.",
        "Architecture is the art and technique of designing and building, as distinguished from the skills associated with construction.",
        "The solar system consists of the Sun and everything gravitationally bound to it, including the eight planets and their moons.",
    ]

    n_samples = st.slider("Number of samples per test", min_value=2, max_value=8, value=8)
    max_tokens = st.slider("Max new tokens (watermarked)", min_value=50, max_value=200, value=200, step=10)

    if st.button("Run experiments", type="primary"):
        gen = WatermarkGenerator(key=key_bytes, lambda_=lambda_, model=model, tokenizer=tokenizer)
        det = WatermarkDetector(key=key_bytes, lambda_=lambda_, model=model, tokenizer=tokenizer)

        # ── Soundness ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Soundness — Human-Written Text")
        human_scores = []
        fp = 0
        soundness_rows = []

        progress = st.progress(0)
        for i, text in enumerate(HUMAN_TEXTS[:n_samples]):
            with st.spinner(f"Testing human text {i+1}/{n_samples}..."):
                score, detected, _ = det.score(text)
                human_scores.append(score)
                if detected:
                    fp += 1
                n_tok = len(tokenizer.encode(text))
                thresh = det.null_mean_per_token * n_tok + det.lambda_ * math.sqrt(n_tok)
                soundness_rows.append({
                    "Text": text[:70] + "...",
                    "Score": f"{score:.2f}",
                    "Threshold": f"{thresh:.2f}",
                    "Watermark Detected": "Yes" if detected else "No",
                })
            progress.progress((i + 1) / n_samples)

        st.table(soundness_rows)
        fpr = fp / n_samples
        if fpr == 0:
            st.success(f"False positive rate: {fpr:.0%}  ({fp}/{n_samples})")
        else:
            st.warning(f"False positive rate: {fpr:.0%}  ({fp}/{n_samples})")

        # ── Completeness ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Completeness — Watermarked Text")
        wm_scores = []
        tp = 0
        completeness_rows = []

        progress2 = st.progress(0)
        for i, prompt in enumerate(PROMPTS[:n_samples]):
            with st.spinner(f"Generating + detecting watermarked text {i+1}/{n_samples}..."):
                result = gen.generate(prompt, max_new_tokens=max_tokens)
                score, detected, best_pos = det.score(result["text"])
                wm_scores.append(score)
                if detected:
                    tp += 1
                n_tok = len(tokenizer.encode(result["text"]))
                thresh = det.null_mean_per_token * n_tok + det.lambda_ * math.sqrt(n_tok)
                completeness_rows.append({
                    "Prompt": prompt[:60] + "...",
                    "Score": f"{score:.2f}",
                    "Threshold": f"{thresh:.2f}",
                    "Entropy": f"{result['entropy_reached']:.1f} bits",
                    "Watermark Detected": "Yes" if detected else "No",
                })
            progress2.progress((i + 1) / n_samples)

        st.table(completeness_rows)
        tdr = tp / n_samples
        if tdr == 1.0:
            st.success(f"True detection rate: {tdr:.0%}  ({tp}/{n_samples})")
        else:
            st.warning(f"True detection rate: {tdr:.0%}  ({tp}/{n_samples})")

        # ── Score distribution plot ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Score Distributions")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        all_scores = human_scores + wm_scores
        bins = np.linspace(min(all_scores) - 5, max(all_scores) + 5, 20)

        axes[0].hist(human_scores, bins=bins, alpha=0.7, label="Human text", color="#4477aa", density=True)
        axes[0].hist(wm_scores,    bins=bins, alpha=0.7, label="Watermarked", color="#c0392b", density=True)
        axes[0].set_xlabel("Detection Score")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Score Histogram")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        bp = axes[1].boxplot(
            [human_scores, wm_scores],
            labels=["Human", "Watermarked"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#4477aa")
        bp["boxes"][1].set_facecolor("#c0392b")
        for p in bp["boxes"]:
            p.set_alpha(0.7)
        axes[1].set_ylabel("Detection Score")
        axes[1].set_title("Score Comparison")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"Undetectable LLM Watermark · λ={lambda_} · n={n_samples}",
            fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Summary metrics ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Human mean score",       f"{np.mean(human_scores):.1f}")
        c2.metric("Watermarked mean score", f"{np.mean(wm_scores):.1f}")
        c3.metric("Score gap",              f"{np.mean(wm_scores) - np.mean(human_scores):.1f}")
        c4.metric("False positive rate",    f"{fpr:.0%}")
