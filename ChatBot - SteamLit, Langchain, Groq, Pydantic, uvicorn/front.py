#Set StreamLit UI
#Connect with Backend

import streamlit as st

st.set_page_config(page_title = "LangGraph Agent Chatbot", layout = 'wide')
st.title("AI Chatbot")
st.write("Interact with me")

system_prompt = st.text_area("Define your AI Agent :", height = 80, placeholder = "Type Here...")

ALLOWEDMODELS=["mixtral-8x7b-32768", 
               "llama-3.3-70b-versatile"]

provider = st.radio("Model Selector",("Groq"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Models",ALLOWEDMODELS)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Query is :", height = 80, placeholder = "Type Query...")

API_URL = 'http://127.0.0.1:9999/chat'

import requests

if st.button("Get to Work Agent"):
    if user_query.strip():

        payload = {'model_name':selected_model,
        'model_provider':provider,
        'system_prompt':system_prompt,
        'messages':[user_query],
        'allow_search':allow_web_search}
            
        response = requests.post(API_URL,json=payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data['error'])
            else:
                st.subheader("Agent Response")
                st.markdown(f"Response :{response_data}")



