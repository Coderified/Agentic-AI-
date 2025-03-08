# Pydantic model - schema validation
# AI for frot emd request
from pydantic import BaseModel
from typing import List


class ReqState(BaseModel):
    model_name:str
    model_provider:str
    system_prompt:str
    messages:List[str]
    allow_search:bool


from fastapi import FastAPI
from ai_agent import get_response_from_query

ALLOWEDMODELS=["llama3-70b-8192", 
               "mixtral-8x7b-32768", 
               "llama-3.3-70b-versatile"
]


app = FastAPI(title="AI Chatbot")

@app.post("/chat")
def chat_endpoint(request:ReqState):
    '''
    API ENdpoint to interact with the Chatbot using LangGraph and Search Tools
    Dynmically selects the model specified in the request
    '''
    if request.model_name not in ALLOWEDMODELS:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    
    model_name = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider


    
    response = get_response_from_query(model_name,query,allow_search,system_prompt,provider)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port = 9999)





