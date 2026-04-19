import torch
import streamlit as st
import tiktoken
import plotly.graph_objects as go
from model import load_model
from config import GPT_CONFIG_124M, MAX_GEN_TOKENS
from sampling import sample_next_token, get_filtered_probs

st.set_page_config(page_title="GPT-2 Playground", page_icon="🤖", layout="wide")
st.title("GPT-2 Playground")
st.caption("Step-by-step autoregressive text generation with GPT-2 124M")

# ------ Load resources ------
@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding('gpt2')

@st.cache_resource
def get_model(cfg):
    return load_model(cfg)

tokenizer = load_tokenizer()
model = get_model(GPT_CONFIG_124M)

# ------ Session state ------
defaults = {
    'generated_ids': [],
    'start_ids': None,
    'internals': None,
    'top_k': 5,
    'top_p': 0.9,
    'min_p': 0.05,
    'temperature': 0.8,
    'use_temperature': False,
    'sampling_mode': 'top-k',
    'masked_ids': [],
    'user_input': "The mesmerizing north light is",
    'last_token': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------ Tabs ------
tab_gen, tab_viz = st.tabs(["Generate", "Visualizations"])

# ==================== Generate Tab ====================
with tab_gen:
    text = st.text_area("Input text:", st.session_state.user_input, height=80)
    if text != st.session_state.user_input:
        st.session_state.user_input = text
        st.session_state.start_ids = None
        st.session_state.generated_ids = []
        st.session_state.internals = None
        st.session_state.masked_ids = []
        st.session_state.last_token = None
        st.rerun()

    if st.session_state.user_input and st.session_state.start_ids is None:
        ids = tokenizer.encode(st.session_state.user_input)
        st.session_state.start_ids = torch.tensor(ids).unsqueeze(0)
        st.session_state.masked_ids = [False] * len(ids)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        if st.session_state.start_ids is not None:
            with st.expander("Mask Input Tokens", expanded=True):
                token_list = st.session_state.start_ids.squeeze(0).tolist()
                token_strings = [tokenizer.decode([tid]) for tid in token_list]
                cols = st.columns(len(token_strings))
                for i, (col, tok) in enumerate(zip(cols, token_strings)):
                    with col:
                        st.session_state.masked_ids[i] = st.checkbox(
                            f"'{tok}'", value=st.session_state.masked_ids[i], key=f"mask_{i}"
                        )

        # Build and display token sequence
        if st.session_state.start_ids is not None:
            full_ids = st.session_state.start_ids.clone()
            if st.session_state.generated_ids:
                full_ids = torch.cat(
                    [full_ids, torch.tensor(st.session_state.generated_ids).unsqueeze(0)], dim=1
                )
            n_input = st.session_state.start_ids.shape[1]
            all_tokens = [tokenizer.decode([tid]) for tid in full_ids.squeeze(0)]
            input_part = " ".join([f"`{t}`" for t in all_tokens[:n_input]])
            gen_part = " ".join([f"**`{t}`**" for t in all_tokens[n_input:]])
            st.write("**Tokens:** " + input_part + (" → " + gen_part if gen_part else ""))

        st.subheader("Generated Text")
        if st.session_state.start_ids is not None:
            full_text = tokenizer.decode(full_ids.squeeze(0).tolist())
        else:
            full_text = st.session_state.user_input
        st.text_area("", full_text, height=150, label_visibility="collapsed")

        if st.session_state.last_token:
            st.success(f"Last generated token: `{st.session_state.last_token}`")

    with col_right:
        st.subheader("Sampling Controls")

        st.session_state.sampling_mode = st.radio(
            "Mode", ["top-k", "top-p", "min-p"],
            index=["top-k", "top-p", "min-p"].index(st.session_state.sampling_mode),
            horizontal=True
        )
        if st.session_state.sampling_mode == "top-k":
            st.session_state.top_k = st.slider("Top-K", min_value=1, max_value=20, value=st.session_state.top_k)
        elif st.session_state.sampling_mode == "top-p":
            st.session_state.top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=st.session_state.top_p, step=0.05)
        elif st.session_state.sampling_mode == "min-p":
            st.session_state.min_p = st.slider("Min-P", min_value=0.01, max_value=0.2, value=st.session_state.min_p, step=0.01)

        st.session_state.use_temperature = st.toggle("Use Temperature", value=st.session_state.use_temperature)
        if st.session_state.use_temperature:
            st.session_state.temperature = st.slider(
                "Temperature", min_value=0.1, max_value=10.0,
                value=st.session_state.temperature, step=0.1
            )

        st.divider()

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate = st.button("Generate ▶", type="primary", use_container_width=True)
        with col_btn2:
            restart = st.button("Restart", use_container_width=True)

        if restart:
            for key in ['generated_ids', 'internals', 'start_ids', 'masked_ids', 'last_token']:
                st.session_state[key] = [] if key in ('generated_ids', 'masked_ids') else None
            st.rerun()

        if generate and st.session_state.start_ids is not None:
            if len(st.session_state.generated_ids) < MAX_GEN_TOKENS:
                masked_input = st.session_state.start_ids.clone()
                for i, masked in enumerate(st.session_state.masked_ids):
                    if masked:
                        masked_input[0, i] = tokenizer.encode("[...]")[0]
                if st.session_state.generated_ids:
                    prev = torch.tensor(st.session_state.generated_ids, dtype=torch.long).unsqueeze(0)
                    masked_input = torch.cat([masked_input, prev], dim=1)

                with torch.no_grad():
                    internals = model.forward_with_internals(masked_input)
                st.session_state.internals = internals

                next_token_id = sample_next_token(
                    logits=internals['logits'][0, -1],
                    mode=st.session_state.sampling_mode,
                    top_k=st.session_state.top_k,
                    top_p=st.session_state.top_p,
                    min_p=st.session_state.min_p,
                    temperature=st.session_state.temperature,
                    use_temperature=st.session_state.use_temperature
                )
                st.session_state.generated_ids.append(next_token_id)
                st.session_state.last_token = tokenizer.decode([next_token_id])
                st.rerun()
            else:
                st.warning(f"Reached {MAX_GEN_TOKENS} token limit. Click Restart.")

# ==================== Visualizations Tab ====================
with tab_viz:
    if st.session_state.internals is None:
        st.info("Generate at least one token to see probability distributions.")
    else:
        logits = st.session_state.internals['logits'][0, -1]
        mode = st.session_state.sampling_mode
        temperature = st.session_state.temperature

        def make_prob_chart(indices, probs, title, color):
            labels = [
                tokenizer.decode([int(i)]).replace('\n', '\\n').strip() or f"[{int(i)}]"
                for i in indices
            ]
            labels = [l[:15] for l in labels]
            token_ids = [int(i) for i in indices]
            p = probs.tolist()

            fig = go.Figure(go.Bar(
                x=p,
                y=labels,
                orientation='h',
                text=[f"{v:.3f}" for v in p],
                textposition='outside',
                marker_color=color,
                hovertemplate="<b>%{y}</b><br>Token ID: %{customdata}<br>Prob: %{x:.4f}<extra></extra>",
                customdata=token_ids,
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Probability",
                yaxis=dict(autorange='reversed'),
                height=max(350, len(indices) * 28),
                margin=dict(l=10, r=80, t=50, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            return fig

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            indices, probs = get_filtered_probs(
                logits=logits, mode=mode,
                top_k=st.session_state.top_k,
                top_p=st.session_state.top_p,
                min_p=st.session_state.min_p,
                temperature=temperature if st.session_state.use_temperature else 1.0
            )
            st.metric("Tokens in sampling pool", len(indices))
            st.plotly_chart(
                make_prob_chart(indices, probs, f"Sampling pool — {mode}", '#4C9BE8'),
                use_container_width=True
            )

        with col_v2:
            indices2, probs2 = get_filtered_probs(
                logits=logits, mode=mode,
                top_k=st.session_state.top_k,
                top_p=st.session_state.top_p,
                min_p=st.session_state.min_p,
                temperature=temperature
            )
            st.metric("Tokens in pool with Temperature", len(indices2))
            st.plotly_chart(
                make_prob_chart(indices2, probs2, f"With Temperature T={temperature}", '#E8854C'),
                use_container_width=True
            )
