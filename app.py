"""
app.py — Streamlit UI for Undetectable LLM Watermarking
Course project SPC-FEH 2026, HSG
"""

import math
import streamlit as st
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from watermark import WatermarkGenerator, DEFAULT_KEY, DEFAULT_LAMBDA
from detect import WatermarkDetector

MAX_NEW_TOKENS = 200

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE        = "#2563EB"   # primary blue — Phase 2 text, buttons, detected bar
BLUE_DARK   = "#1E3A8A"   # threshold line, hover
BLUE_TINT   = "#EFF6FF"   # card / sidebar backgrounds
BLUE_BORDER = "#BFDBFE"   # card borders
GRAY_PHASE1 = "#6B7280"   # Phase 1 text — neutral gray, clearly distinct from blue
GRAY_BAR    = "#CBD5E1"   # not-detected bar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Undetectable LLM Watermarking", layout="wide")

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: #FFFFFF; }}

        section[data-testid="stSidebar"] {{
            border-right: 1px solid {BLUE_BORDER};
        }}

        div[data-testid="metric-container"] {{
            background-color: {BLUE_TINT};
            border: 1px solid {BLUE_BORDER};
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }}

        .stButton > button[kind="primary"] {{
            background-color: {BLUE};
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {BLUE_DARK};
            border: none;
        }}
        .stButton > button[kind="secondary"] {{
            border: 2px solid {BLUE_BORDER};
            color: {BLUE};
            border-radius: 8px;
            font-weight: 600;
        }}

        hr {{ border-color: {BLUE_BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading GPT-2 (one-time, ~500 MB)...")
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("LLM Watermarking")
st.sidebar.caption("Undetectable watermarks via a cryptographic PRF.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["1 · Generate", "2 · Detect"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Parameters**")
secret_key = st.sidebar.text_input("Secret key", value="spc_feh_2026_watermark_key")
key_bytes = secret_key.encode()
lambda_ = DEFAULT_LAMBDA


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
if page == "1 · Generate":
    st.title("Watermarked Text Generation")
    st.markdown(
        f"Enter a prompt and generate watermarked text. Tokens are colour-coded by phase:  \n"
        f"<span style='color:{GRAY_PHASE1}; font-weight:600;'>&#9632; Gray — Phase 1</span> &nbsp; natural sampling &nbsp;&nbsp;"
        f"<span style='color:{BLUE}; font-weight:600;'>&#9632; Blue — Phase 2</span> &nbsp; PRF-guided (watermarked)",
        unsafe_allow_html=True,
    )

    prompt = st.text_input(
        "Prompt",
        value="The history of cryptography begins in ancient civilisations",
    )

    if st.button("Generate watermarked text", type="primary"):
        with st.spinner("Generating..."):
            gen = WatermarkGenerator(key=key_bytes, lambda_=lambda_, model=model, tokenizer=tokenizer)
            result = gen.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)

        col1, col2, col3 = st.columns(3)
        col1.metric("Entropy at seed lock", f"{result['entropy_reached']:.2f} nats")
        col2.metric("Seed locked at token", result["seed_position"] if result["seed_position"] else "never")
        col3.metric("Tokens generated", len(result["generated_tokens"]))

        st.markdown("---")

        if result["seed_position"] is None:
            st.warning("Seed never locked — entropy threshold not reached.")
            st.write(result["text"])
        else:
            seed_pos = result["seed_position"]
            phase1_text = tokenizer.decode(result["generated_tokens"][:seed_pos], skip_special_tokens=True)
            phase2_text = tokenizer.decode(result["generated_tokens"][seed_pos:], skip_special_tokens=True)

            st.markdown("#### Generated text")
            st.markdown(
                f'<div style="font-size:1.05rem; line-height:2.0; padding:1.4rem 1.6rem; '
                f'background:#FFFFFF; border-radius:10px; border:1px solid {BLUE_BORDER}; '
                f'box-shadow:0 1px 4px rgba(37,99,235,0.08);">'
                f'<span style="color:{GRAY_PHASE1};">{phase1_text} </span>'
                f'<span style="color:{BLUE}; font-weight:500;">{phase2_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DETECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2 · Detect":
    st.title("Detection")
    st.markdown(
        "Paste any text below. The detector tries every prefix position as a candidate seed "
        "and reports the best alignment score."
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
            label = "correct key" if run_correct else "wrong key"

            with st.spinner(f"Running detector ({label})..."):
                det = WatermarkDetector(key=used_key, lambda_=lambda_, model=model, tokenizer=tokenizer)
                score, detected, seed_pos = det.score(text_input)
                n = len(tokenizer.encode(text_input))
                remaining = max(n - seed_pos - 1, 1) if seed_pos >= 0 else n
                threshold = det.null_mean_per_token * remaining + det.lambda_ * math.sqrt(remaining)

            st.markdown("---")

            if detected:
                st.success(f"Watermark detected  ·  {label}")
            else:
                st.info(f"No watermark detected  ·  {label}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Score", f"{score:.2f}")
            col2.metric("Threshold", f"{threshold:.2f}")
            col3.metric("Gap", f"{score - threshold:+.2f}")
            col4.metric("Best seed position", seed_pos if seed_pos >= 0 else "n/a")

            st.markdown("#### Score vs Threshold")
            fig, ax = plt.subplots(figsize=(8, 1.4))
            ax.barh(["Score"], [score], color=BLUE if detected else GRAY_BAR, height=0.45)
            ax.axvline(threshold, color=BLUE_DARK, linewidth=2, linestyle="--",
                       label=f"Threshold ({threshold:.1f})")
            ax.set_xlim(0, max(score, threshold) * 1.2)
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlabel("Score", color="#374151")
            ax.tick_params(colors="#374151")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color(BLUE_BORDER)
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
