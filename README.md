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
  An interactive tool for understanding how language models generate text — built from scratch in PyTorch.
</p>

<p align="center">
  <a href="https://taanmaay-gpt-2-playground.hf.space"><img src="https://img.shields.io/badge/🤗 Live Demo-HF Spaces-blue" /></a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
</p>

---

![GPT-2 Playground walkthrough](https://raw.githubusercontent.com/Tanmay-jam/GPT-2-Playground/assets/basicwalkthrough.gif)

---

## Pages

### 🤖 Next Token Generation
Watch GPT-2 generate text one token at a time. Control every aspect of sampling and observe the probability distribution update live.

- Generate step-by-step or auto-generate with live chart updates
- Three sampling modes: **Top-K**, **Top-P (nucleus)**, **Min-P** with temperature scaling
- Mask any input token to see how context shapes predictions
- **Visualizations tab:** token confidence history and entropy per step across the full generation sequence

### 🔒 Constrained Decoding
Restrict the token pool to a chosen subset and see how it reshapes the probability distribution at each step.

- Four constraint modes: digits only, alphabetic only, custom vocabulary, forbidden words
- Side-by-side charts: full distribution (forbidden tokens greyed out) vs constrained distribution (renormalized)
- Demonstrates that constrained decoding is a decoding-level technique — no retraining required

## Architecture

No HuggingFace `transformers`. Every component is implemented from scratch.

| File | Contents |
|---|---|
| `blocks.py` | `LayerNorm`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock` |
| `model.py` | `GPTModel`, `forward_with_internals` |
| `sampling.py` | `sample_next_token`, `get_filtered_probs` — top-k / top-p / min-p |
| `config.py` | `GPT_CONFIG_124M` |
| `app.py` | Multipage router (`st.navigation`) |
| `pages/` | One file per page |

GPT-2 124M weights are downloaded from [taanmaay/GPT-2-124M-weights](https://huggingface.co/taanmaay/GPT-2-124M-weights) at Docker build time.

## Run locally

```bash
git clone https://github.com/Tanmay-jam/GPT-2-Playground
cd GPT-2-Playground
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
