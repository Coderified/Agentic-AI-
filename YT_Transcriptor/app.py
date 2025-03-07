import streamlit as st
from dotenv import load_dotenv
load_dotenv() #loads the envmt variables
import os,requests
import google.generativeai as gai

from youtube_transcript_api import YouTubeTranscriptApi

gai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

prompt='''
You are a YT summarizer and summarize it within 20 words. 
'''


##STEPS 
# 1) Extract transcript from the YT video
# 2) Feed into the generate_context func to get summary 

def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        print(video_id)
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e


def generate_context(transcript):
    groq_api_url = "https://api.groq.com/v1/summarize"  # Replace with actual Groq API endpoint
    headers = {
       "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b",  # Choose the model
        "prompt": f"Summarize the following YouTube transcript in 3-5 bullet points:\n{transcript}"
    }
    response = requests.post(groq_api_url, headers=headers, json=data)
    return response.text



st.title("YT Transcript to Detialed Notes")

yt_link = st.text_input("Enter YT Link : ")

if yt_link:
    video_id = yt_link.split('=')[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get the Notes"):
    yt_transcript = extract_transcript_details(yt_link)

    if yt_transcript:
        summary = generate_context(yt_transcript)
        st.markdown("Notes here ")

        st.write(summary)




