import torch
import streamlit as st
import tiktoken
import plotly.graph_objects as go
from model import load_model, DraftGPTModel
from config import GPT_CONFIG_124M
from speculative import run_speculative_decoding

st.set_page_config(page_title="Speculative Decoding", page_icon="⚡", layout="wide")
st.title("⚡ Speculative Decoding")
st.caption("Generating multiple tokens with a single target model forward pass")

# ------ Explanation ──────────────────────────────────────────────────────────
with st.expander("How it works", expanded=True):
    st.markdown("""
**The problem with standard autoregressive decoding:**
Each token requires one full forward pass through the target model (12 layers, 124M params).
Generating 10 tokens = 10 sequential forward passes.

**Speculative decoding solution:**
1. A cheap **draft model** (fewer layers, same weights) generates K tokens greedily and fast
2. The **target model** (full 12 layers) verifies all K tokens in **one parallel forward pass**
3. Each draft token is **accepted** with probability `min(1, p_target / p_draft)`
4. At the first rejection, the token is **resampled** from an adjusted distribution and generation stops
5. If all K accepted — a **bonus token** is sampled from the target for free

**Result:** Multiple tokens generated with far fewer target model calls — same output distribution as pure target sampling.
""")

# ------ Load resources ───────────────────────────────────────────────────────
@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding('gpt2')

@st.cache_resource
def get_target_model():
    return load_model(GPT_CONFIG_124M)

tokenizer = load_tokenizer()
target_model = get_target_model()

# ------ Controls ─────────────────────────────────────────────────────────────
col_input, col_controls = st.columns([3, 2])

with col_input:
    user_input = st.text_area(
        "Input prompt:",
        value="The mesmerizing north light is",
        height=80
    )

with col_controls:
    K = st.slider("K — draft tokens to generate", min_value=1, max_value=10, value=5)
    n_draft_layers = st.slider("Draft model layers (out of 12)", min_value=1, max_value=11, value=6)
    use_temperature = st.toggle("Use Temperature", value=False)
    temperature = st.slider("Temperature", min_value=0.1, max_value=5.0, value=1.0, step=0.1) if use_temperature else 1.0
    run = st.button("▶ Run Speculative Decoding", type="primary", use_container_width=True)

# ------ Run ──────────────────────────────────────────────────────────────────
if run and user_input:
    draft_model = DraftGPTModel(target_model, n_layers=n_draft_layers)
    draft_model.eval()

    input_ids = torch.tensor(tokenizer.encode(user_input)).unsqueeze(0)

    with st.spinner(f"Draft model ({n_draft_layers} layers) generating {K} tokens, target verifying..."):
        with torch.no_grad():
            steps, accepted_ids = run_speculative_decoding(
                target_model=target_model,
                draft_model=draft_model,
                input_ids=input_ids,
                tokenizer=tokenizer,
                K=K,
                temperature=temperature
            )

    n_accepted = sum(s.accepted for s in steps)
    bonus = len(accepted_ids) > n_accepted

    # ── Summary metrics ───────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Draft tokens generated", K)
    m2.metric("Tokens accepted", n_accepted)
    m3.metric("Total tokens produced", len(accepted_ids))
    m4.metric("Target forward passes", 1, help="Standard autoregressive would need " + str(len(accepted_ids)))

    # ── Generated text ────────────────────────────────────────────────────
    st.subheader("Result")
    accepted_text = tokenizer.decode(accepted_ids)
    full_text = user_input + accepted_text
    st.text_area("", full_text, height=100, label_visibility="collapsed")

    input_tokens = " ".join([f"`{tokenizer.decode([t])}`" for t in tokenizer.encode(user_input)])
    accepted_tokens = " ".join([f"**`{tokenizer.decode([t])}`**" for t in accepted_ids])
    st.write("**Tokens:** " + input_tokens + " → " + accepted_tokens)

    if bonus:
        st.info(f"All {K} draft tokens accepted — bonus token `{tokenizer.decode([accepted_ids[-1]])}` sampled from target.")

    # ── Step-by-step verification table ──────────────────────────────────
    st.divider()
    st.subheader("Draft → Verify")

    col_table, col_chart = st.columns([3, 2])

    with col_table:
        for i, step in enumerate(steps):
            accepted_label = "✅ Accepted" if step.accepted else "❌ Rejected"
            color = "#1a472a" if step.accepted else "#4a1010"
            border = "#2ecc71" if step.accepted else "#e74c3c"

            resampled_note = ""
            if not step.accepted and step.resampled_token_text:
                resampled_note = f"<br><small>↳ Resampled: <b>`{step.resampled_token_text}`</b></small>"

            st.markdown(f"""
<div style="background:{color}; border-left: 4px solid {border}; padding:10px 14px; margin-bottom:8px; border-radius:4px;">
  <b>Token {i+1}:</b> <code>{step.token_text}</code> &nbsp; {accepted_label}<br>
  Draft prob: <b>{step.draft_prob:.4f}</b> &nbsp;|&nbsp;
  Target prob: <b>{step.target_prob:.4f}</b> &nbsp;|&nbsp;
  Acceptance: <b>{step.acceptance_prob:.2%}</b>
  {resampled_note}
</div>
""", unsafe_allow_html=True)

    with col_chart:
        # Bar chart comparing draft vs target probs
        labels = [f"T{i+1}: {s.token_text[:8]}" for i, s in enumerate(steps)]
        draft_ps = [s.draft_prob for s in steps]
        target_ps = [s.target_prob for s in steps]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Draft prob", x=labels, y=draft_ps, marker_color='#5B9BD5'))
        fig.add_trace(go.Bar(name="Target prob", x=labels, y=target_ps, marker_color='#ED7D31'))
        fig.update_layout(
            barmode='group',
            title="Draft vs Target Probabilities",
            yaxis_title="Probability",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=50, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Acceptance probability bar
        accept_colors = ['#2ecc71' if s.accepted else '#e74c3c' for s in steps]
        fig2 = go.Figure(go.Bar(
            x=labels,
            y=[s.acceptance_prob for s in steps],
            marker_color=accept_colors,
            text=[f"{s.acceptance_prob:.0%}" for s in steps],
            textposition='outside',
        ))
        fig2.update_layout(
            title="Acceptance Probability per Token",
            yaxis=dict(title="min(1, p_target / p_draft)", range=[0, 1.2]),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=60)
        )
        st.plotly_chart(fig2, use_container_width=True)

elif not run:
    st.info("Configure the parameters above and click **▶ Run Speculative Decoding** to see the process.")
