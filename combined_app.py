import streamlit as st
import vector_rag
import graph_rag

st.set_page_config(page_title="RAG App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select RAG Type", ["Vector RAG", "Graph RAG"])

if page == "Vector RAG":
    vector_rag.main()
elif page == "Graph RAG":
    graph_rag.main()