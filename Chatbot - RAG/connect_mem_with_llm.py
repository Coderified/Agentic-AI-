# create llm , connnect with faiss and create chain
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

HF_TOKEN = os.enivron.get("HF_TOKEN")

hf_repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(hf_repo_id):
    llm=HuggingFaceEndpoint(repo_id=hf_repo_id, 
                            temperature=0.5, 
                            model_kwargs={"token":HF_TOKEN,
                                          "max_length":'512'}
                            )
    return llm    


customr_prompt_template = '''
Use only the input information to answer user's question.
Don't try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

No small talk please.
'''

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(custom_prompt_template,input_variables={"context","question"})
    return prompt


## LOADING THE DWTW FROM THE DATABASE PATH

DB_FAISS_PATH = "vectordatabase/db_faiss"
embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH,embed_model,allow_dangerous_deserialization=True)


## Create a chain 

chain = RetrievalQA.from_chain_type()