import streamlit as st
from openai import OpenAI
from google.cloud import firestore
from google.oauth2 import service_account
from datetime import datetime 

st.set_page_config (page_titile="Data Analytics with Python Support", layout="centered")

#1. Initialize Firebase Firestore Connection - for logging analytics
@st.cache_resource
def get_database_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return firestore.Client(credentials=creds)

db = get_database_client ()

#2. Gate the App with Student first name (this will be a pseudonym assigned number)
if "student_first_name" not in st.session_state:
    st.title("Welcome to Datta")
    st.write("Please enter your first name to begin the session. ")

    student_first_name_input = st.text_input("Student First Name:", placeholder="e.g., Joe").strip()
    