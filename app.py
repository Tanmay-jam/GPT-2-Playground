import torch
import torch.nn as nn
import streamlit as st
import tiktoken
import matplotlib.pyplot as plt

from blocks import LayerNorm, TransformerBlock

st.set_page_config(layout="wide")

# ------ GPTModel Definition (with internals extraction) ------
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward_with_internals(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        positions = torch.arange(seq_len, device=in_idx.device).unsqueeze(0)
        pos_embeds = self.pos_emb(positions)
        input_embeds = tok_embeds + pos_embeds
        dropped = self.drop_emb(input_embeds)

        internals = {
            'tokens': None,
            'token_ids': in_idx.squeeze(0).tolist(),
            'embeddings': input_embeds.detach().cpu(),
            'positional_embeddings': pos_embeds.detach().cpu(),
            'layer_outputs': [],
            'logits': None
        }

        x = dropped
        for block in self.trf_blocks:
            x = block(x)
            internals['layer_outputs'].append(x.detach().cpu())

        normed = self.final_norm(x)
        logits = self.out_head(normed)
        internals['logits'] = logits.detach().cpu()

        return internals

# ------ Utility Functions ------
@st.cache_resource
def load_tokenizer(model_name: str = 'gpt2'):
    return tiktoken.get_encoding('gpt2')

@st.cache_resource
def load_model(cfg, weights_path: str):
    model = GPTModel(cfg)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

# ------ Streamlit App ------
st.title("GPT-2 Playground")

# Load resources
tokenizer = load_tokenizer()
CFG = GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}
WEIGHTS_PATH = 'D:\\IIT Patna\\Project_LFS\\gpt_pytorch_weights.pth'
model = load_model(CFG, WEIGHTS_PATH)

# Initialize session state
if 'generated_ids' not in st.session_state:
    st.session_state.generated_ids = []
if 'start_ids' not in st.session_state:
    st.session_state.start_ids = None
if 'internals' not in st.session_state:
    st.session_state.internals = None
if 'top_k' not in st.session_state:
    st.session_state.top_k = 5
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.8
if 'use_temperature' not in st.session_state:
    st.session_state.use_temperature = False
if 'masked_ids' not in st.session_state:
    st.session_state.masked_ids = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = "Every effort moves you"

# Text input
text = st.text_area("Enter text input:", st.session_state.user_input)
if text != st.session_state.user_input:
    st.session_state.user_input = text
    st.session_state.start_ids = None
    st.session_state.generated_ids = []
    st.session_state.internals = None
    st.session_state.masked_ids = []
    st.rerun()

# Restart button
if st.button("Restart Generation"):
    st.session_state.generated_ids = []
    st.session_state.internals = None
    st.session_state.start_ids = None
    st.session_state.masked_ids = []
    st.rerun()

cola, colb, colc = st.columns(3)
with cola:
    st.session_state.top_k = st.slider("Top-K Tokens",min_value=1,max_value=20,value=st.session_state.top_k)
with colb:
    st.session_state.temperature = st.slider("Temperature",min_value=0.0,max_value=10.0,value=st.session_state.temperature,step=0.1)
with colc:
    st.session_state.use_temperature = st.radio("Use Temperature for Generation", [False, True], index=int(st.session_state.use_temperature))

if st.session_state.user_input and st.session_state.start_ids is None:
    start_token_ids = tokenizer.encode(st.session_state.user_input)
    input_ids = torch.tensor(start_token_ids).unsqueeze(0)
    st.session_state.start_ids = input_ids
    st.session_state.masked_ids = [False] * input_ids.shape[1]

# Masking UI
if st.session_state.start_ids is not None:
    st.subheader("🔲 Mask Input Tokens")
    token_list = st.session_state.start_ids.squeeze(0).tolist()
    token_strings = [tokenizer.decode([tid]) for tid in token_list]
    cols = st.columns(len(token_strings))
    for i, (col, tok) in enumerate(zip(cols, token_strings)):
        with col:
            st.session_state.masked_ids[i] = st.checkbox(f"'{tok}'", value=st.session_state.masked_ids[i], key=f"mask_{i}")

# Generate one token at a time
if st.session_state.start_ids is not None:
    col1, col2 = st.columns(2)
    with col1:
        full_ids = st.session_state.start_ids.clone()
        if st.session_state.generated_ids:
            gen_tensor = torch.tensor(st.session_state.generated_ids).unsqueeze(0)
            full_ids = torch.cat([full_ids, gen_tensor], dim=1)
        input_tokens = [tokenizer.decode([tid]) for tid in full_ids.squeeze(0)]
        st.write("Tokens passed to model:")
        st.write(input_tokens)

        # Show embeddings with masking
        masked_ids = st.session_state.masked_ids
        display_ids = st.session_state.start_ids.squeeze(0).tolist()
        for i, is_masked in enumerate(masked_ids):
            if is_masked:
                display_ids[i] = tokenizer.encode("[...]")[0]
        display_tensor = torch.tensor(display_ids).unsqueeze(0)
        if st.session_state.generated_ids:
            display_tensor = torch.cat([display_tensor, torch.tensor(st.session_state.generated_ids).unsqueeze(0)], dim=1)
        with torch.no_grad():
            model_output = model.forward_with_internals(display_tensor)
        embeddings = model_output['embeddings'][:, :len(display_ids), :]
        st.write("Embeddings (masked if selected):")
        st.write(embeddings)

    with col2:
        if st.button("Generate Next Token"):
            if len(st.session_state.generated_ids) < 20:
                prev_ids = st.session_state.generated_ids
                full_ids = st.session_state.start_ids.clone()
                for i, masked in enumerate(st.session_state.masked_ids):
                    if masked:
                        full_ids[0, i] = tokenizer.encode("[...]")[0]

                if prev_ids:
                    prev_tensor = torch.tensor(prev_ids, dtype=torch.long).unsqueeze(0)
                    full_ids = torch.cat([full_ids, prev_tensor], dim=1)

                with torch.no_grad():
                    internals = model.forward_with_internals(full_ids)
                st.session_state.internals = internals

                last_logits = internals['logits'][0, -1]
                if st.session_state.use_temperature:
                    scaled_logits = last_logits / st.session_state.temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                else:
                    probs = torch.softmax(last_logits, dim=-1)

                topk = torch.topk(probs, k=st.session_state.top_k)
                sampled_index = torch.multinomial(topk.values, num_samples=1)
                next_token_id = topk.indices[sampled_index].item()
                st.session_state.generated_ids.append(next_token_id)
                predicted_token = tokenizer.decode([next_token_id])
                st.success(f"Generated token: `{predicted_token}`")
            else:
                st.warning("Maximum generation limit (20 tokens) reached. Click 'Restart Generation' above to start over.")

        full_ids = st.session_state.start_ids.clone()
        if st.session_state.generated_ids:
            gen_tensor = torch.tensor(st.session_state.generated_ids).unsqueeze(0)
            full_ids = torch.cat([full_ids, gen_tensor], dim=1)

        full_text = tokenizer.decode(full_ids.squeeze(0).tolist())
        st.text_area("Generated Text So Far", full_text, height=200)

        if st.session_state.internals is not None:
            with st.expander(f"Logits (Top {st.session_state.top_k})"):
                first_logits = st.session_state.internals['logits'][0, -1]
                topk = torch.topk(first_logits, k=st.session_state.top_k)
                probs = torch.softmax(topk.values, dim=-1)
                st.write("Top Token IDs + Probabilities:")
                token_strings = [tokenizer.decode([int(i)]) for i in topk.indices]
                fig, ax = plt.subplots()
                bars1=ax.bar(token_strings, probs.tolist(), color='skyblue')
                ax.bar_label(bars1, labels=topk.indices.tolist(), label_type='center', color='white', fontsize=12)
                ax.set_ylabel("Probability")
                ax.set_title(f"Top-{st.session_state.top_k} Predicted Tokens")
                st.pyplot(fig)

            with st.expander(f"Temperature-scaled Logits (Top {st.session_state.top_k})"):
                temperature = st.session_state.temperature
                scaled_logits = st.session_state.internals['logits'][0, -1] / temperature
                scaled_topk = torch.topk(scaled_logits, k=st.session_state.top_k)
                scaled_probs = torch.softmax(scaled_topk.values, dim=-1)
                st.write(f"Top Token IDs + Probabilities (Temp={temperature}):")
                scaled_tokens = [tokenizer.decode([int(i)]) for i in scaled_topk.indices]
                fig2, ax2 = plt.subplots()
                bars2=ax2.bar(scaled_tokens, scaled_probs.tolist(), color='salmon')
                ax2.bar_label(bars2, labels=scaled_topk.indices.tolist(), label_type='center', color='white', fontsize=12)
                ax2.set_ylabel("Probability")
                ax2.set_title(f"Top-{st.session_state.top_k} Tokens with Temperature Scaling (T={temperature})")
                st.pyplot(fig2)
