import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

FAISS_DATAPATH = "vectordatabase/db_faiss"

@st.cache_resource 

# This is a Streamlit decorator used to cache the result of a function so that it doesn’t need to be recomputed on every rerun of the app.
# Why Use It?:
# Loading a vector store (especially a large one) can be computationally expensive and time-consuming.
# By caching the result, the vector store is loaded only once, and subsequent calls to the function return the cached result, improving performance.

# When Does It Run?:
# The function is executed the first time it is called, and its result is cached.

# On subsequent calls, the cached result is returned instead of re-executing the function.


def get_vector_store():
    embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(FAISS_DATAPATH,embed_model,allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

HF_TOKEN = os.environ.get("HF_TOKEN")
def load_llm(hf_repo_id):
    llm=HuggingFaceEndpoint(repo_id=hf_repo_id, 
                            temperature=0.5, 
                            model_kwargs={"token":HF_TOKEN,
                                          "max_length":'512'}
                            )
    return llm    


def main():
    st.set_page_config(page_title = "Chatbot", layout = 'wide')
    
    st.markdown(
    """
    <style>
   
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/abstract-equalizer-particles-waves-background_23-2148185706.jpg?t=st=1741527414~exp=1741531014~hmac=5937577747088318a2bec6b7cd90685ed41fc67c70f5f44b48416c1f9cd11bbd&w=1380");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    h1 {
        color: white !important; /* Force white text color */
        text-align: center; /* Center the title */
    }
    .stChatMessage {
        background-color: rgba(0, 0,0, 0.98); /* Semi-transparent white background */
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        }
    .stChatMessage p {
        color: white; /* White text color */
    }

    </style>
    """,
    unsafe_allow_html=True
)
    st.title("PDF AI Chatbot - Give me PDF, & Shoot the Questions")
    if 'messages' not in st.session_state:
        st.session_state.messages =  []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    

    
    prompt = st.chat_input("Pass your Prompt")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role":'user',"content":prompt})

        custom_prompt_template = '''
                Use only the input information to answer user's question.
                Don't try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                No small talk please.
                '''
        
        HF_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'



        try:
            vectorstore = get_vector_store()
            if vectorstore is None:
                st.error("Failed to load Vector Store")

            chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HF_REPO_ID),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k':3}), #how many similar top documents that will be taken into 
                    return_source_documents=True,
                    chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
                )


            response = chain.invoke({'query':prompt}) # user input prompt goes into in the custom_prompt_template as "question" and 
                                                        #retriever fills the answer in context

                #The connection between the user input (prompt) and the {question} placeholder is handled automatically 
                # The RetrievalQA chain:
                # Takes the user's question (prompt) as input --> # Retrieves relevant documents (context).
                # Populates the custom prompt template with {context} and {question} --> # Passes the final prompt to the LLM for processing.

            result = response["result"]
            source_docs = response["source_documents"]

            
            st.chat_message('assistant').markdown(result + "\n \n Source Documents Pages related to this:" +
                                                    ", ".join(map(str, (x.metadata['page'] for x in source_docs))))
            #st.chat_message('assistant').markdown(result + "\n \n Source Documents :" + str(source_docs))
            st.session_state.messages.append({"role":'assistant',"content":response})


        except Exception as e:
            st.error(f"Error :{str(e)}")    

if __name__ == "__main__":
    main()