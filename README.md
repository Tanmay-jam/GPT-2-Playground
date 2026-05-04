---
title: GPT 2 Playground
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<h1 align="center">GPT-2 Playground</h1>

<p align="center">
  Step-by-step autoregressive text generation with GPT-2 124M — built from scratch in PyTorch.
</p>

<p align="center">
  <a href="https://taanmaay-gpt-2-playground.hf.space"><img src="https://img.shields.io/badge/🤗 Live Demo-HF Spaces-blue" /></a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
</p>

---

## What it does

Watch GPT-2 generate text one token at a time. Control every aspect of sampling, observe the probability distribution update live, and understand why the model picks each token.

## Features

**Generate tab**
- Generate tokens one at a time or auto-generate the full sequence with live chart updates
- Three sampling modes: **Top-K**, **Top-P (nucleus)**, **Min-P**
- Temperature scaling toggle
- Mask any input token to see how context shapes predictions

**Visualizations tab**
- **Sampling distribution** — probability bar chart for the current sampling pool, chosen token highlighted
- **Token confidence history** — raw softmax probability of each chosen token over the generation sequence
- **Entropy per step** — Shannon entropy of the full distribution at each step; reveals where the model was uncertain

## Architecture

No HuggingFace `transformers`. Every component is implemented from scratch.

| File | Contents |
|---|---|
| `blocks.py` | `LayerNorm`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock` |
| `model.py` | `GPTModel`, `forward_with_internals` |
| `sampling.py` | `sample_next_token`, `get_filtered_probs` — top-k / top-p / min-p |
| `config.py` | `GPT_CONFIG_124M` |
| `app.py` | Streamlit UI |

GPT-2 124M weights are downloaded from [taanmaay/GPT-2-124M-weights](https://huggingface.co/taanmaay/GPT-2-124M-weights) at Docker build time.

## Run locally

```bash
git clone https://github.com/Tanmay-jam/GPT-2-Playground
cd GPT-2-Playground
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
