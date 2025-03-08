#Setup API Keys
#Setup LLM & Tools
#Setup AI Agent
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")

from langchain_groq import ChatGroq

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import create_react_agent


def get_response_from_query(model_name,query,allow_search,system_prompt,provider):
    if provider=='Groq':
        llm = ChatGroq(model=model_name)

    if allow_search:
        tools=[TavilySearchResults(max_results = 2)] 
    else:
        tools=[]


    agent = create_react_agent(model=llm, 
                           tools = tools,
                           prompt = system_prompt)

    response = agent.invoke({"messages":query})
    messages = response.get("messages")
    ai_messages = [x.content for x in messages if isinstance(x,AIMessage)]
    return (ai_messages[-1])