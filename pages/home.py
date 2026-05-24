import streamlit as st

st.title("GPT-2 Playground")
st.markdown("An interactive tool for understanding how language models generate text — built from scratch in PyTorch with GPT-2 124M.")

st.divider()

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 🤖 Next Token Generation")
        st.markdown(
            "Watch GPT-2 generate text one token at a time.\n\n"
            "Experiment with **Top-K**, **Top-P**, and **Min-P** sampling, "
            "temperature scaling, and token masking. See the probability "
            "distribution update live after each step."
        )
        st.markdown("**Concepts:** autoregressive decoding, sampling, temperature")
        if st.button("Explore", key="nav_gen", use_container_width=True, type="primary"):
            st.switch_page("pages/next_token_generation.py")

with col2:
    with st.container(border=True):
        st.markdown("### 🔒 Constrained Decoding")
        st.markdown(
            "Restrict the token pool to a chosen subset — digits only, "
            "alphabetic only, a custom vocabulary, or block specific words.\n\n"
            "See how masking forbidden tokens reshapes the probability "
            "distribution and forces structure at the decoding level."
        )
        st.markdown("**Concepts:** constrained decoding, logit masking, BPE tokens")
        if st.button("Explore", key="nav_con", use_container_width=True, type="primary"):
            st.switch_page("pages/constrained_decoding.py")

with col3:
    with st.container(border=True):
        st.markdown("### ⚡ Speculative Decoding")
        st.markdown(
            "A small draft model proposes K tokens. The full target model "
            "verifies all K in **one parallel forward pass** — accepting or "
            "rejecting each to match the exact target distribution.\n\n"
            "Fewer target model calls, same output quality."
        )
        st.markdown("**Concepts:** speculative decoding, draft-verify, acceptance sampling")
        st.button("Work in Progress", disabled=True, use_container_width=True)
