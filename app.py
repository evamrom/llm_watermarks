"""
app.py - Streamlit UI for Undetectable LLM Watermarking
Course project SPC-FEH 2026, HSG
"""

import math
import streamlit as st
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from watermark import WatermarkGenerator, DEFAULT_KEY, DEFAULT_LAMBDA
from detect import WatermarkDetector
from substringwatermark import SubstringWatermarkGenerator
from substringdetect import SubstringWatermarkDetector

MAX_NEW_TOKENS = 200

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE        = "#2563EB"   # primary blue - Phase 2 text, buttons, detected bar
BLUE_DARK   = "#1E3A8A"   # threshold line, hover
BLUE_TINT   = "#EFF6FF"   # card / sidebar backgrounds
BLUE_BORDER = "#BFDBFE"   # card borders
GRAY_PHASE1 = "#6B7280"   # Phase 1 text - neutral gray, clearly distinct from blue
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


def detection_threshold(detector: WatermarkDetector, text: str, seed_bit_pos: int) -> float:
    if seed_bit_pos < 0:
        return 0.0

    tokens = detector.tokenizer.encode(text)
    length_bits = len(detector._tokens_to_bits(tokens))
    remaining_bits = max(length_bits - seed_bit_pos, 0)
    return remaining_bits + detector.lambda_ * math.sqrt(remaining_bits)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("LLM Watermarking")
st.sidebar.caption("Undetectable watermarks via a cryptographic PRF.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["1 · Generate", "2 · Detect", "3 · Substring"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Parameters**")
secret_key = st.sidebar.text_input("Secret key", value="spc_feh_2026_watermark_key")
key_bytes = secret_key.encode()
lambda_ = st.sidebar.slider(
    "Lambda",
    min_value=4.0,
    max_value=80.0,
    value=float(DEFAULT_LAMBDA),
    step=4.0,
    help="Higher values delay seed lock, making the gray Phase 1 segment longer but leaving fewer watermarked tokens.",
)
max_new_tokens = st.sidebar.slider(
    "Max new tokens",
    min_value=80,
    max_value=400,
    value=MAX_NEW_TOKENS,
    step=20,
    help="Generate more tokens when Lambda is large so Phase 2 still has enough text for detection.",
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - GENERATE
# ══════════════════════════════════════════════════════════════════════════════
if page == "1 · Generate":
    st.title("Watermarked Text Generation")
    st.markdown(
        f"Enter a prompt and generate watermarked text. Tokens are colour-coded by phase:  \n"
        f"<span style='color:{GRAY_PHASE1}; font-weight:600;'>&#9632; Gray - Phase 1</span> &nbsp; natural sampling &nbsp;&nbsp;"
        f"<span style='color:{BLUE}; font-weight:600;'>&#9632; Blue - Phase 2</span> &nbsp; PRF-guided (watermarked)",
        unsafe_allow_html=True,
    )

    prompt = st.text_input(
        "Prompt",
        value="The history of cryptography begins in ancient civilisations",
    )

    if st.button("Generate watermarked text", type="primary"):
        with st.spinner("Generating..."):
            gen = WatermarkGenerator(key=key_bytes, lambda_=lambda_, model=model, tokenizer=tokenizer)
            result = gen.generate(prompt, max_new_tokens=max_new_tokens)
        st.session_state["generated_text"] = result["text"]
        st.session_state["generate_result"] = result
        st.session_state["generate_num_bits"] = gen.num_bits

    if "generate_result" in st.session_state:
        result = st.session_state["generate_result"]
        num_bits = st.session_state["generate_num_bits"]
        seed_bit_length = result["seed_bit_length"]
        seed_token_count = math.ceil(seed_bit_length / num_bits) if seed_bit_length else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Entropy at seed lock", f"{result['entropy_reached']:.2f} nats")
        col2.metric(
            "Seed lock position",
            f"{seed_bit_length} bits (~{seed_token_count} tokens)" if seed_bit_length else "never",
        )
        col3.metric("Tokens generated", len(result["generated_tokens"]))

        st.markdown("---")

        if seed_bit_length is None:
            st.warning("Seed never locked - entropy threshold not reached.")
            st.write(result["text"])
        else:
            seed_pos = seed_token_count
            phase1_text = tokenizer.decode(result["generated_tokens"][:seed_pos], skip_special_tokens=True)
            phase2_text = tokenizer.decode(result["generated_tokens"][seed_pos:], skip_special_tokens=True)

            st.markdown("#### Generated text")
            st.markdown(
                f'<div style="font-size:1.05rem; line-height:2.0; padding:1.4rem 1.6rem; '
                f'background:#FFFFFF; border-radius:10px; border:1px solid {BLUE_BORDER}; '
                f'box-shadow:0 1px 4px rgba(37,99,235,0.08);">'
                f'<span style="color:{GRAY_PHASE1};">{phase1_text}</span>'
                f'<span style="color:{BLUE}; font-weight:500;">{phase2_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("#### Algorithm internals")

            n_phase2_tokens = len(result["generated_tokens"]) - seed_token_count
            ci1, ci2, ci3 = st.columns(3)
            ci1.metric("Phase 1 tokens shown", seed_token_count)
            ci2.metric("Phase 2 tokens (PRF-guided)", n_phase2_tokens)
            ci3.metric("PRF calls made", len(result["prf_values"]))

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**Entropy accumulation - Phase 1**")
                st.caption(
                    "Each bit sampled naturally contributes −log p to cumulative entropy. "
                    "Once it hits λ the seed locks and Phase 2 begins."
                )
                if result["entropy_history"]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    xs = list(range(len(result["entropy_history"])))
                    ys = result["entropy_history"]
                    ax.plot(xs, ys, color=GRAY_PHASE1, linewidth=1.8)
                    ax.fill_between(xs, ys, alpha=0.12, color=GRAY_PHASE1)
                    ax.axhline(lambda_, color=BLUE, linewidth=1.8, linestyle="--",
                               label=f"λ = {lambda_} (seed lock)")
                    ax.scatter([len(xs) - 1], [ys[-1]], color=BLUE, zorder=5, s=50)
                    ax.set_xlabel("Bit index (Phase 1)", fontsize=9)
                    ax.set_ylabel("Cumulative entropy (nats)", fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    fig.patch.set_facecolor("#FFFFFF")
                    ax.set_facecolor("#FFFFFF")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            with chart_col2:
                st.markdown("**PRF u-values - Phase 2**")
                st.caption(
                    "The PRF outputs u ∈ [0, 1] used to choose each bit. "
                    "A uniform distribution proves the output statistics are unchanged - "
                    "the watermark is undetectable without the key."
                )
                if result["prf_values"]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(result["prf_values"], bins=20, color=BLUE, alpha=0.7,
                            density=True, label="Observed u-values")
                    ax.axhline(1.0, color=GRAY_PHASE1, linewidth=1.8, linestyle="--",
                               label="Expected (uniform)")
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("u = F_sk(r, position)", fontsize=9)
                    ax.set_ylabel("Density", fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    fig.patch.set_facecolor("#FFFFFF")
                    ax.set_facecolor("#FFFFFF")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No PRF calls - all phase-2 bits were degenerate (p=0 or p=1).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - DETECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2 · Detect":
    st.title("Detection")
    st.markdown(
        "Paste any text below. For text generated in this app, detection uses the saved token bits "
        "directly; external pasted text is tokenized from the string."
    )

    generated_text = st.session_state.get("generated_text", "")
    text_input = st.text_area(
        "Text to analyse",
        value=generated_text,
        height=200,
        placeholder="Paste text here - watermarked or human-written...",
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
                det = WatermarkDetector(key=used_key, lambda_=lambda_, tokenizer=tokenizer)
                use_saved_bits = (
                    "generate_result" in st.session_state
                    and text_input == generated_text
                )
                if use_saved_bits:
                    det_result = det.score_details(
                        bits=st.session_state["generate_result"]["generated_bits"]
                    )
                    source_label = "saved generated bits"
                else:
                    det_result = det.score_details(text=text_input)
                    source_label = "tokenized string"

            st.session_state["det_result"] = det_result
            st.session_state["det_label"]  = label
            st.session_state["det_source"] = source_label

    if "det_result" in st.session_state:
        det_result = st.session_state["det_result"]
        label      = st.session_state["det_label"]
        source     = st.session_state.get("det_source", "tokenized string")
        score     = det_result["score"]
        detected  = det_result["detected"]
        seed_pos  = det_result["seed_bit_pos"]
        threshold = det_result["threshold"]

        st.markdown("---")

        if detected:
            st.success(f"Watermark detected  ·  {label}")
        else:
            st.info(f"No watermark detected  ·  {label}")
        st.caption(f"Detection source: {source}")

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

        # ── Cumulative score trajectory ───────────────────────────────────
        if det_result["trajectory"]:
            st.markdown("---")
            st.markdown("#### Score trajectory over bits")
            st.caption(
                "For watermarked text the score (blue) climbs above the threshold (dashed). "
                "For human text or the wrong key, it stays flat - a random walk around the threshold line."
            )
            traj  = det_result["trajectory"]
            thr_t = det_result["traj_thresholds"]
            tok_xs = [x / 16 for x in range(len(traj))]

            fig2, ax2 = plt.subplots(figsize=(10, 3))
            score_color = BLUE if detected else GRAY_BAR
            ax2.plot(tok_xs, traj,  color=score_color, linewidth=1.5, label="Cumulative score")
            ax2.plot(tok_xs, thr_t, color=BLUE_DARK, linewidth=1.5, linestyle="--",
                     label=f"Threshold  (n + λ√n,  λ={lambda_})")
            above = [s > t for s, t in zip(traj, thr_t)]
            ax2.fill_between(tok_xs, traj, thr_t, where=above,
                             alpha=0.15, color=BLUE, label="Score > threshold")
            ax2.set_xlabel("Tokens after seed lock", fontsize=9)
            ax2.set_ylabel("Score", fontsize=9)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            fig2.patch.set_facecolor("#FFFFFF")
            ax2.set_facecolor("#FFFFFF")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Seed prefix search ────────────────────────────────────────────
        if det_result["all_results"]:
            st.markdown("#### Seed prefix search")
            st.caption(
                "The detector tries every possible seed length and scores the remaining bits. "
                "Points above zero mean detection; the best-scoring prefix (marked) is used."
            )
            positions = [r["seed_bit_pos"] / 16 for r in det_result["all_results"]]
            excesses  = [r["score"] - r["threshold"] for r in det_result["all_results"]]

            fig3, ax3 = plt.subplots(figsize=(10, 2.5))
            ax3.plot(positions, excesses, color=GRAY_BAR, linewidth=1.0, alpha=0.8)
            ax3.axhline(0, color=BLUE_DARK, linewidth=1.5, linestyle="--",
                        label="Detection boundary (excess = 0)")
            ax3.fill_between(positions, excesses, 0,
                             where=[e > 0 for e in excesses],
                             alpha=0.15, color=BLUE)
            if seed_pos >= 0:
                best_excess = score - threshold
                ax3.scatter([seed_pos / 16], [best_excess], color=BLUE if detected else GRAY_BAR,
                            zorder=5, s=70, label=f"Best seed - token {seed_pos // 16}")
            ax3.set_xlabel("Candidate seed length (tokens)", fontsize=9)
            ax3.set_ylabel("Score − Threshold", fontsize=9)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            fig3.patch.set_facecolor("#FFFFFF")
            ax3.set_facecolor("#FFFFFF")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - SUBSTRING COMPLETE WATERMARK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "3 · Substring":
    st.title("Substring-Complete Watermarking")
    st.markdown(
        "Generate with rolling seed blocks, then remove the prefix and detect a middle substring."
    )

    prompt = st.text_input(
        "Prompt",
        value="The history of cryptography begins in ancient civilisations",
        key="substring_prompt",
    )

    max_tokens = st.slider(
        "Generation length",
        min_value=120,
        max_value=360,
        value=max(120, min(360, max_new_tokens)),
        step=20,
    )

    if st.button("Generate substring-complete text", type="primary"):
        with st.spinner("Generating rolling-seed watermark..."):
            gen = SubstringWatermarkGenerator(
                key=key_bytes,
                lambda_=lambda_,
                model=model,
                tokenizer=tokenizer,
            )
            result = gen.generate(prompt, max_new_tokens=max_tokens)
        st.session_state["substring_result"] = result
        st.session_state["substring_num_bits"] = gen.num_bits

    if "substring_result" in st.session_state:
        result = st.session_state["substring_result"]
        num_bits = st.session_state["substring_num_bits"]
        tokens = result["generated_tokens"]
        bits = result["generated_bits"]
        seed_events = result["seed_events"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Generated tokens", len(tokens))
        col2.metric("Seed blocks", len(seed_events))
        col3.metric("PRF calls", len(result["prf_values"]))

        st.markdown("#### Generated continuation")
        st.markdown(
            f'<div style="font-size:1.05rem; line-height:2.0; padding:1.4rem 1.6rem; '
            f'background:#FFFFFF; border-radius:10px; border:1px solid {BLUE_BORDER}; '
            f'box-shadow:0 1px 4px rgba(37,99,235,0.08);">'
            f'{result["text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if seed_events:
            st.markdown("#### Seed updates")
            st.dataframe(
                [
                    {
                        "kind": event["kind"],
                        "start token": event["seed_start_token"],
                        "length tokens": event["seed_token_length"],
                        "entropy": round(event["entropy"], 2),
                        "threshold": round(event["threshold"], 2),
                    }
                    for event in seed_events
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No seed block was locked. Generate more tokens or lower λ.")

        if len(tokens) >= 80:
            st.markdown("---")
            st.markdown("#### Truncated substring test")

            default_start = min(len(tokens) // 3, max(len(tokens) - 64, 0))
            start_token = st.slider(
                "Substring start token",
                min_value=0,
                max_value=max(len(tokens) - 32, 0),
                value=default_start,
                step=1,
            )
            max_len = len(tokens) - start_token
            length_token = st.slider(
                "Substring length",
                min_value=min(32, max_len),
                max_value=max_len,
                value=max(32, max_len),
                step=1,
            )

            sub_tokens = tokens[start_token:start_token + length_token]
            sub_bits = bits[start_token * num_bits:(start_token + length_token) * num_bits]
            sub_text = tokenizer.decode(sub_tokens, skip_special_tokens=True)
            substring_slice_key = (start_token, length_token, len(tokens))

            st.text_area(
                "Selected substring",
                value=sub_text,
                height=160,
                disabled=True,
            )

            col_a, col_b = st.columns(2)
            run_correct = col_a.button("Detect substring (correct key)", type="primary")
            run_wrong = col_b.button("Detect substring (wrong key)", type="secondary")

            if run_correct or run_wrong:
                used_key = key_bytes if run_correct else b"wrong_key_xyz"
                label = "correct key" if run_correct else "wrong key"
                with st.spinner(f"Running substring detector ({label})..."):
                    det = SubstringWatermarkDetector(key=used_key, lambda_=lambda_, tokenizer=tokenizer)
                    det_result = det.score_details(bits=sub_bits)
                st.session_state["substring_det_result"] = det_result
                st.session_state["substring_det_label"] = label
                st.session_state["substring_det_slice_key"] = substring_slice_key

        current_slice_key = locals().get("substring_slice_key")
        has_current_det_result = (
            "substring_det_result" in st.session_state
            and st.session_state.get("substring_det_slice_key") == current_slice_key
        )
        if has_current_det_result:
            det_result = st.session_state["substring_det_result"]
            label = st.session_state["substring_det_label"]
            score = det_result["score"]
            threshold = det_result["threshold"]
            detected = det_result["detected"]

            st.markdown("---")
            if detected:
                st.success(f"Substring watermark detected  ·  {label}")
            else:
                st.info(f"No substring watermark detected  ·  {label}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", f"{score:.2f}")
            c2.metric("Threshold", f"{threshold:.2f}")
            c3.metric("Gap", f"{score - threshold:+.2f}")
            c4.metric("Seed window", f"{det_result['seed_bit_pos'] // num_bits}+{det_result['seed_bit_length'] // num_bits} tok")

            st.markdown("#### Best substring match")
            fig, ax = plt.subplots(figsize=(8, 1.4))
            ax.barh(["Score"], [score], color=BLUE if detected else GRAY_BAR, height=0.45)
            ax.axvline(threshold, color=BLUE_DARK, linewidth=2, linestyle="--",
                       label=f"Threshold ({threshold:.1f})")
            ax.set_xlim(0, max(score, threshold, 1.0) * 1.2)
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

            if det_result["all_results"]:
                positions = [r["seed_bit_pos"] / num_bits for r in det_result["all_results"]]
                excesses = [r["score"] - r["threshold"] for r in det_result["all_results"]]

                fig2, ax2 = plt.subplots(figsize=(10, 2.5))
                ax2.plot(positions, excesses, color=GRAY_BAR, linewidth=1.0, alpha=0.75)
                ax2.axhline(0, color=BLUE_DARK, linewidth=1.5, linestyle="--",
                            label="Detection boundary")
                ax2.fill_between(positions, excesses, 0,
                                 where=[e > 0 for e in excesses],
                                 alpha=0.15, color=BLUE)
                ax2.set_xlabel("Candidate seed start inside substring (tokens)", fontsize=9)
                ax2.set_ylabel("Score - Threshold", fontsize=9)
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
                fig2.patch.set_facecolor("#FFFFFF")
                ax2.set_facecolor("#FFFFFF")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
