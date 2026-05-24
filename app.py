import streamlit as st

st.set_page_config(page_title="GPT-2 Playground", page_icon="🤖", layout="wide")

pg = st.navigation([
    st.Page("pages/home.py",                  title="Home",                  icon="🏠"),
    st.Page("pages/next_token_generation.py", title="Next Token Generation", icon="🤖"),
    st.Page("pages/constrained_decoding.py",  title="Constrained Decoding",  icon="🔒"),
])
pg.run()
