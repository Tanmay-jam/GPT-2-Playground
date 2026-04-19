---
title: GPT 2 Playground
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# GPT-2 Playground

An interactive app for exploring GPT-2 text generation step by step — built from scratch in PyTorch.

🔗 **Live Demo:** https://taanmaay-gpt-2-playground.hf.space

📦 **Model Weights:** https://huggingface.co/taanmaay/GPT-2-124M-weights

---

## Features

**Generation**
- Enter any prompt and generate tokens one at a time
- Three sampling strategies: **Top-K**, **Top-P (nucleus)**, **Min-P**
- Temperature scaling with live toggle
- Mask input tokens to observe how context affects predictions

**Visualizations**
- Interactive probability distribution charts (Plotly) for the sampling pool
- Side-by-side comparison: raw distribution vs temperature-scaled
- Selected token highlighted in green
- **Perplexity** of the current sequence displayed as a metric

---

## Architecture

Built entirely from scratch — no HuggingFace `transformers`, no prebuilt model APIs.

| File | Contents |
|---|---|
| `blocks.py` | `LayerNorm`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock` |
| `model.py` | `GPTModel` class, weight loading |
| `sampling.py` | `sample_next_token`, `get_filtered_probs` — top-k, top-p, min-p |
| `config.py` | `GPT_CONFIG_124M`, `MAX_GEN_TOKENS` |
| `app.py` | Streamlit UI |

GPT-2 124M weights loaded from Hugging Face Hub at Docker build time.

---

## Run Locally

```bash
git clone https://github.com/Tanmay-jam/GPT-2-Playground
cd GPT-2-Playground
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
