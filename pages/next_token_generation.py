import time
import torch
import streamlit as st
import tiktoken
import plotly.graph_objects as go
from model import load_model
from config import GPT_CONFIG_124M
from sampling import sample_next_token, get_filtered_probs

st.title("Next Token Generation")
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
    'generated_probs': [],
    'generated_entropies': [],
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
    'max_gen_tokens': 50,
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
        st.session_state.generated_probs = []
        st.session_state.generated_entropies = []
        st.session_state.internals = None
        st.session_state.masked_ids = []
        st.session_state.last_token = None
        st.rerun()

    if st.session_state.user_input and st.session_state.start_ids is None:
        ids = tokenizer.encode(st.session_state.user_input)
        st.session_state.start_ids = torch.tensor(ids).unsqueeze(0)
        st.session_state.masked_ids = [False] * len(ids)

    EOS_TOKEN_ID = 50256

    def run_one_step():
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
        raw_probs = torch.softmax(internals['logits'][0, -1], dim=-1)
        st.session_state.generated_probs.append(raw_probs[next_token_id].item())
        entropy = -(raw_probs * torch.log(raw_probs + 1e-10)).sum().item()
        st.session_state.generated_entropies.append(entropy)
        return next_token_id

    def make_chart_fig(internals, last_token_id):
        logits = internals['logits'][0, -1]
        indices, probs = get_filtered_probs(
            logits=logits,
            mode=st.session_state.sampling_mode,
            top_k=st.session_state.top_k,
            top_p=st.session_state.top_p,
            min_p=st.session_state.min_p,
            temperature=st.session_state.temperature if st.session_state.use_temperature else 1.0
        )
        labels = [
            tokenizer.decode([int(i)]).replace('\n', '\\n').strip() or f"[{int(i)}]"
            for i in indices
        ]
        labels = [l[:15] for l in labels]
        colors = [
            '#2ecc71' if (last_token_id is not None and int(idx) == last_token_id) else '#4C9BE8'
            for idx in indices
        ]
        fig = go.Figure(go.Bar(
            x=probs.tolist(), y=labels, orientation='h',
            text=[f"{v:.3f}" for v in probs.tolist()],
            textposition='outside',
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Token ID: %{customdata}<br>Prob: %{x:.4f}<extra></extra>",
            customdata=[int(i) for i in indices],
        ))
        fig.update_layout(
            xaxis_title="Probability",
            yaxis=dict(autorange='reversed'),
            height=max(250, len(indices) * 24),
            margin=dict(l=10, r=70, t=20, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

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
        text_ph = st.empty()
        full_text = tokenizer.decode(full_ids.squeeze(0).tolist()) if st.session_state.start_ids is not None else st.session_state.user_input
        text_ph.text_area("", full_text, height=150, label_visibility="collapsed")

        if st.session_state.last_token:
            st.success(f"Last generated token: `{st.session_state.last_token}`")

        st.subheader("Sampling Distribution")
        chart_ph = st.empty()
        if st.session_state.internals is not None:
            last_token_id = st.session_state.generated_ids[-1] if st.session_state.generated_ids else None
            chart_ph.plotly_chart(make_chart_fig(st.session_state.internals, last_token_id), use_container_width=True)

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

        st.session_state.max_gen_tokens = st.slider(
            "Max tokens to generate", min_value=10, max_value=200,
            value=st.session_state.max_gen_tokens, step=10
        )

        st.divider()

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            generate = st.button("Generate ▶", type="primary", use_container_width=True)
        with col_btn2:
            auto_generate = st.button("Auto Generate ⚡", use_container_width=True)
        with col_btn3:
            restart = st.button("Restart", use_container_width=True)

        if restart:
            for key in ['generated_ids', 'generated_probs', 'generated_entropies', 'internals', 'start_ids', 'masked_ids', 'last_token']:
                st.session_state[key] = [] if key in ('generated_ids', 'generated_probs', 'generated_entropies', 'masked_ids') else None
            st.rerun()

        if generate and st.session_state.start_ids is not None:
            if len(st.session_state.generated_ids) < st.session_state.max_gen_tokens:
                run_one_step()
                st.rerun()
            else:
                st.warning(f"Reached {st.session_state.max_gen_tokens} token limit. Click Restart.")

    # Auto-generate loop — runs after both columns so it can update col_left placeholders
    if auto_generate and st.session_state.start_ids is not None:
        remaining = st.session_state.max_gen_tokens - len(st.session_state.generated_ids)
        for _ in range(remaining):
            next_token_id = run_one_step()
            current_ids = st.session_state.start_ids.clone()
            if st.session_state.generated_ids:
                current_ids = torch.cat(
                    [current_ids, torch.tensor(st.session_state.generated_ids).unsqueeze(0)], dim=1
                )
            text_ph.text_area("", tokenizer.decode(current_ids.squeeze(0).tolist()),
                              height=150, label_visibility="collapsed")
            last_id = st.session_state.generated_ids[-1]
            chart_ph.plotly_chart(make_chart_fig(st.session_state.internals, last_id), use_container_width=True)
            if next_token_id == EOS_TOKEN_ID:
                break
            time.sleep(1.0)
        st.rerun()

# ==================== Visualizations Tab ====================
with tab_viz:
    if st.session_state.internals is None:
        st.info("Generate at least one token to see visualizations.")
    else:
        internals = st.session_state.internals

        st.subheader("Token Confidence History")
        st.caption("Raw softmax probability assigned to each chosen token at generation time — shows where the model was confident vs. uncertain.")

        if st.session_state.generated_ids:
            gen_tokens = [
                tokenizer.decode([tid]).replace('\n', '\\n').strip() or f"[{tid}]"
                for tid in st.session_state.generated_ids
            ]
            gen_tokens = [t[:12] for t in gen_tokens]
            probs_history = st.session_state.generated_probs
            mean_prob = sum(probs_history) / len(probs_history)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=list(range(len(gen_tokens))),
                y=probs_history,
                text=[f"{p:.3f}" for p in probs_history],
                textposition='outside',
                marker_color=['#2ecc71' if p >= mean_prob else '#E8854C' for p in probs_history],
                hovertemplate="<b>%{customdata}</b><br>Step: %{x}<br>Prob: %{y:.4f}<extra></extra>",
                customdata=gen_tokens,
            ))
            fig_hist.add_hline(
                y=mean_prob, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                annotation_text=f"mean {mean_prob:.3f}", annotation_position="top right"
            )
            fig_hist.update_layout(
                xaxis=dict(tickmode='array', tickvals=list(range(len(gen_tokens))), ticktext=gen_tokens, tickangle=-45),
                yaxis_title="Probability",
                height=320,
                margin=dict(l=10, r=20, t=20, b=80),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Generate some tokens to see confidence history.")

        st.divider()

        st.subheader("Entropy per Step")
        st.caption("Shannon entropy of the full token distribution at each generation step (nats). High entropy = many plausible next tokens. Low entropy = model was confident.")

        if st.session_state.generated_entropies:
            entropies = st.session_state.generated_entropies
            mean_ent = sum(entropies) / len(entropies)
            step_labels = [
                tokenizer.decode([tid]).replace('\n', '\\n').strip() or f"[{tid}]"
                for tid in st.session_state.generated_ids
            ]
            step_labels = [t[:10] for t in step_labels]

            fig_ent = go.Figure(go.Bar(
                x=list(range(len(entropies))),
                y=entropies,
                marker_color=['#E8854C' if e >= mean_ent else '#4C9BE8' for e in entropies],
                text=[f"{e:.2f}" for e in entropies],
                textposition='outside',
                hovertemplate="<b>%{customdata}</b><br>Step %{x}<br>Entropy: %{y:.3f} nats<extra></extra>",
                customdata=step_labels,
            ))
            fig_ent.add_hline(
                y=mean_ent, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                annotation_text=f"mean {mean_ent:.2f}", annotation_position="top right"
            )
            fig_ent.update_layout(
                xaxis=dict(tickmode='array', tickvals=list(range(len(step_labels))), ticktext=step_labels, tickangle=-45),
                yaxis_title="Entropy (nats)",
                height=300,
                margin=dict(l=10, r=20, t=20, b=80),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_ent, use_container_width=True)
