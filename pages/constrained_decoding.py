import time
import torch
import streamlit as st
import tiktoken
import plotly.graph_objects as go
from model import load_model
from config import GPT_CONFIG_124M

st.title("🔒 Constrained Decoding")
st.caption("Restricting token sampling to a chosen subset — showing how structure can be enforced at the decoding level")

with st.expander("How it works", expanded=True):
    st.markdown("""
**Standard decoding** samples from the full vocabulary (~50k tokens) at every step based on the model's learned probabilities.

**Constrained decoding** restricts sampling to a subset of allowed token IDs. Mechanically, at each step:
1. The model runs a full forward pass — nothing inside the model changes
2. Logits for **forbidden** tokens are set to `−∞` before softmax
3. The distribution is **renormalized** over allowed tokens only
4. Sampling proceeds normally within that restricted pool

This is entirely a **decoding-level** technique — no retraining required, works on any model.

""")

# ── Resources ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding('gpt2')

@st.cache_resource
def get_model(cfg):
    return load_model(cfg)

tokenizer = load_tokenizer()
model = get_model(GPT_CONFIG_124M)

VOCAB_SIZE = 50257

# ── Precomputed constraint sets ───────────────────────────────────────────────
@st.cache_data
def get_digit_ids(_tok):
    return [
        i for i in range(VOCAB_SIZE)
        if _tok.decode([i]).replace(' ', '').replace(',', '').replace('.', '')
                          .replace('%', '').replace('-', '').isdigit()
    ]

@st.cache_data
def get_alpha_ids(_tok):
    return [
        i for i in range(VOCAB_SIZE)
        if _tok.decode([i]).strip() and all(c.isalpha() or c == ' ' for c in _tok.decode([i]))
    ]

def get_custom_vocab_ids(words_text):
    words = [w.strip() for w in words_text.replace(',', '\n').split('\n') if w.strip()]
    ids = set()
    skipped = []
    for word in words:
        with_space = tokenizer.encode(' ' + word)
        without_space = tokenizer.encode(word)
        if len(with_space) == 1:
            ids.add(with_space[0])
        elif len(without_space) == 1:
            ids.add(without_space[0])
        else:
            skipped.append(word)
    return list(ids), skipped

def get_forbidden_ids(forbidden_text):
    forbidden_words = [w.strip() for w in forbidden_text.replace(',', '\n').split('\n') if w.strip()]
    blocked = set()
    for word in forbidden_words:
        blocked.update(tokenizer.encode(' ' + word))
        blocked.update(tokenizer.encode(word))
    return [i for i in range(VOCAB_SIZE) if i not in blocked]

def apply_constraint(logits, allowed_ids):
    mask = torch.full((VOCAB_SIZE,), float('-inf'))
    mask[torch.tensor(allowed_ids, dtype=torch.long)] = 0.0
    return logits + mask

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    'cd_generated_ids': [],
    'cd_start_ids': None,
    'cd_internals': None,
    'cd_last_token': None,
    'cd_prompt': '',
    'cd_mode': '',
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Example content per mode ──────────────────────────────────────────────────
MODES = ['Digits only', 'Alphabetic only', 'Custom vocabulary', 'Forbidden words']

EXAMPLE_PROMPTS = {
    'Digits only':       "The spacecraft was launched in the year",
    'Alphabetic only':   "The ancient warrior raised his sword and",
    'Custom vocabulary': "The world was full of",
    'Forbidden words':   "The most important thing in life is to",
}

EXAMPLE_CUSTOM_VOCAB = "hope, fear, love, hate, light, dark, war, peace, life, death, truth, shadow, fire, water, sky, earth"
EXAMPLE_FORBIDDEN    = "the, a, an, is, was, are, were, it, be, been, have, has, had"

MODE_DESCRIPTIONS = {
    'Digits only':       "Allows only tokens whose text is purely numeric — digits, decimals, commas, percent signs. Forces the model to generate number sequences.",
    'Alphabetic only':   "Allows only tokens composed of letters and spaces. No punctuation, no numbers. The model generates prose without any special characters.",
    'Custom vocabulary': "You define the allowed words. Only tokens derived from those words are permitted. A minimalist language for the model to work within.",
    'Forbidden words':   "All tokens are allowed except those encoding your specified words. Useful for demonstrating how blocking common words (stopwords) changes generation style.",
}

# ── Controls ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    constraint_mode = st.radio("Constraint mode", MODES, horizontal=True)
    st.caption(MODE_DESCRIPTIONS[constraint_mode])

    prompt = st.text_area(
        "Prompt:",
        value=EXAMPLE_PROMPTS[constraint_mode],
        height=80,
        key=f"cd_prompt_input_{constraint_mode}",
    )

with col_right:
    st.subheader("Constraint Settings")

    custom_vocab_text = EXAMPLE_CUSTOM_VOCAB
    forbidden_text = EXAMPLE_FORBIDDEN

    if constraint_mode == 'Digits only':
        st.success("Precomputed — 1,692 numeric tokens available.")

    elif constraint_mode == 'Alphabetic only':
        st.success("Precomputed — 47,236 alphabetic tokens available.")

    elif constraint_mode == 'Custom vocabulary':
        custom_vocab_text = st.text_area(
            "Allowed words (comma or newline separated):",
            value=EXAMPLE_CUSTOM_VOCAB,
            height=120,
        )
        ids, skipped = get_custom_vocab_ids(custom_vocab_text)
        st.caption(f"{len(ids)} single-token words resolved.")
        if skipped:
            st.warning(f"Skipped (multi-token, would allow sub-word fragments): **{', '.join(skipped)}**")

    elif constraint_mode == 'Forbidden words':
        forbidden_text = st.text_area(
            "Forbidden words (comma or newline separated):",
            value=EXAMPLE_FORBIDDEN,
            height=100,
        )
        n_blocked = VOCAB_SIZE - len(get_forbidden_ids(forbidden_text))
        st.caption(f"{n_blocked} token IDs blocked. {VOCAB_SIZE - n_blocked:,} remaining.")

    max_tokens = st.slider("Max tokens to generate", min_value=5, max_value=60, value=25)

# ── Build allowed ID list for current mode ────────────────────────────────────
def build_allowed_ids():
    if constraint_mode == 'Digits only':
        return get_digit_ids(tokenizer)
    elif constraint_mode == 'Alphabetic only':
        return get_alpha_ids(tokenizer)
    elif constraint_mode == 'Custom vocabulary':
        ids, _ = get_custom_vocab_ids(custom_vocab_text)
        return ids
    else:
        return get_forbidden_ids(forbidden_text)

# ── Reset when prompt or mode changes ─────────────────────────────────────────
st.divider()

col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    run_step  = st.button("Generate Step ▶", type="primary", use_container_width=True)
with col_b2:
    auto_run  = st.button("Auto Generate ⚡", use_container_width=True)
with col_b3:
    reset_btn = st.button("Reset", use_container_width=True)

def reset_state():
    ids = tokenizer.encode(prompt)
    st.session_state.cd_start_ids    = torch.tensor(ids).unsqueeze(0)
    st.session_state.cd_generated_ids = []
    st.session_state.cd_internals     = None
    st.session_state.cd_last_token    = None
    st.session_state.cd_prompt        = prompt
    st.session_state.cd_mode          = constraint_mode

if (reset_btn
        or st.session_state.cd_start_ids is None
        or st.session_state.cd_prompt != prompt
        or st.session_state.cd_mode != constraint_mode):
    reset_state()
    if reset_btn:
        st.rerun()

# ── Placeholders (must be defined before loops that write to them) ─────────────
text_ph = st.empty()

col_chart_full, col_chart_constrained = st.columns(2)

# ── One generation step ───────────────────────────────────────────────────────
def run_step_fn(allowed_ids):
    current_ids = st.session_state.cd_start_ids.clone()
    if st.session_state.cd_generated_ids:
        gen = torch.tensor(st.session_state.cd_generated_ids, dtype=torch.long).unsqueeze(0)
        current_ids = torch.cat([current_ids, gen], dim=1)

    with torch.no_grad():
        internals = model.forward_with_internals(current_ids)

    st.session_state.cd_internals = internals
    logits = internals['logits'][0, -1]

    constrained_logits = apply_constraint(logits, allowed_ids)
    probs = torch.softmax(constrained_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()

    st.session_state.cd_generated_ids.append(next_id)
    st.session_state.cd_last_token = tokenizer.decode([next_id])
    return next_id, logits

# ── Chart builders ────────────────────────────────────────────────────────────
def make_full_fig(logits, allowed_ids, top_n=15):
    full_probs, full_ids = torch.topk(torch.softmax(logits, dim=-1), top_n)
    allowed_set = set(allowed_ids)
    labels = [
        tokenizer.decode([int(i)]).replace('\n', '\\n').strip() or f"[{int(i)}]"
        for i in full_ids
    ]
    labels = [l[:12] for l in labels]
    colors = [
        '#4C9BE8' if int(i) in allowed_set else 'rgba(140,140,140,0.25)'
        for i in full_ids
    ]
    fig = go.Figure(go.Bar(
        x=full_probs.tolist(), y=labels, orientation='h',
        marker_color=colors,
        text=[f"{p:.3f}" for p in full_probs.tolist()],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Prob: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Full distribution — top {top_n} tokens<br><sup>Blue = allowed · Grey = forbidden</sup>",
        xaxis_title="Probability",
        yaxis=dict(autorange='reversed'),
        height=max(320, top_n * 26),
        margin=dict(l=10, r=70, t=55, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def make_constrained_fig(logits, allowed_ids, last_token_id=None, top_n=10):
    constrained = apply_constraint(logits, allowed_ids)
    probs = torch.softmax(constrained, dim=-1)
    k = min(top_n, len(allowed_ids))
    top_probs, top_ids = torch.topk(probs, k)
    labels = [
        tokenizer.decode([int(i)]).replace('\n', '\\n').strip() or f"[{int(i)}]"
        for i in top_ids
    ]
    labels = [l[:12] for l in labels]
    colors = [
        '#2ecc71' if (last_token_id is not None and int(i) == last_token_id) else '#E8854C'
        for i in top_ids
    ]
    fig = go.Figure(go.Bar(
        x=top_probs.tolist(), y=labels, orientation='h',
        marker_color=colors,
        text=[f"{p:.3f}" for p in top_probs.tolist()],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Constrained prob: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Constrained distribution — top {k} allowed tokens<br><sup>Renormalized · Green = chosen</sup>",
        xaxis_title="Probability",
        yaxis=dict(autorange='reversed'),
        height=max(320, k * 26),
        margin=dict(l=10, r=70, t=55, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def render_text():
    full_ids = st.session_state.cd_start_ids.clone()
    if st.session_state.cd_generated_ids:
        gen = torch.tensor(st.session_state.cd_generated_ids, dtype=torch.long).unsqueeze(0)
        full_ids = torch.cat([full_ids, gen], dim=1)
    n_input = st.session_state.cd_start_ids.shape[1]
    all_toks = [tokenizer.decode([t]) for t in full_ids.squeeze(0).tolist()]
    input_part = " ".join([f"`{t}`" for t in all_toks[:n_input]])
    gen_part   = " ".join([f"**`{t}`**" for t in all_toks[n_input:]])
    text_ph.markdown("**Tokens:** " + input_part + (" → " + gen_part if gen_part else ""))

def render_charts(logits, allowed_ids, last_token_id=None):
    with col_chart_full:
        st.plotly_chart(make_full_fig(logits, allowed_ids), use_container_width=True)
    with col_chart_constrained:
        st.plotly_chart(make_constrained_fig(logits, allowed_ids, last_token_id), use_container_width=True)

# ── Render existing state ─────────────────────────────────────────────────────
render_text()

if st.session_state.cd_internals is not None:
    allowed_ids = build_allowed_ids()
    last_id = st.session_state.cd_generated_ids[-1] if st.session_state.cd_generated_ids else None
    render_charts(st.session_state.cd_internals['logits'][0, -1], allowed_ids, last_id)
else:
    with col_chart_full:
        st.info("Run a generation step to see the full probability distribution.")
    with col_chart_constrained:
        st.info("Run a generation step to see the constrained distribution.")

if st.session_state.cd_last_token:
    st.success(f"Last token: `{st.session_state.cd_last_token}`  —  constraint mode: **{constraint_mode}**")

# ── Button handlers ───────────────────────────────────────────────────────────
if run_step:
    if len(st.session_state.cd_generated_ids) < max_tokens:
        allowed_ids = build_allowed_ids()
        run_step_fn(allowed_ids)
        st.rerun()
    else:
        st.warning(f"Reached {max_tokens} token limit. Click Reset.")

if auto_run:
    allowed_ids = build_allowed_ids()
    remaining = max_tokens - len(st.session_state.cd_generated_ids)
    for _ in range(remaining):
        last_id, logits = run_step_fn(allowed_ids)
        render_text()
        render_charts(logits, allowed_ids, last_id)
        time.sleep(1.0)
    st.rerun()
