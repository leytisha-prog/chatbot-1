import streamlit as st

st.set_page_config(
    page_title="Programming for Data Analytics Course Assistant"
    page_icon=":material/code:",
)

st.title("Programming for Data Analytics Course Assistant")
st.write("The Streamlit application is working.")

condition = st.selectbox(
    "Development test condition",
    ["Non-metacognitive", "Metacognitive"],
)

st.success(f"Current text condition: {condition}")