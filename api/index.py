import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.embeddings import Embeddings
from google.genai import types
from fastapi.responses import HTMLResponse
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

app = FastAPI()

GOOGLE_API_KEY           = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY          = os.getenv("SERPAPI_API_KEY")
OPENWEATHERMAP_API_KEY   = os.getenv("OPENWEATHERMAP_API_KEY")

llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(api_version="v1")  
)


class TravelPlan(BaseModel):
    destination: str
    is_feasible: bool
    budget_analysis: str
    suggested_daily_budget: float
    reasoning: str

@tool
def verify_travel_feasibility(plan_details: str) -> str:
    """
    Use this tool to check if a user's budget and destination are realistic.
    Input should be a string describing the destination and budget.
    """
    instruction = """
    You are a Travel Logic Engine. Analyze the request:
    1. Check if budget is realistic for the destination.
    2. If impossible, set is_feasible to False.
    3. Always respond in structured JSON.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{instruction}\n\nPlan to analyze: {plan_details}",
        config={"response_mime_type": "application/json", "response_schema": TravelPlan}
    )
    return response.text


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

private_guide_content = [
    Document(page_content="SECRET DEAL: Use code 'JAMBO2026' for 20% off at any Diani beach resort."),
    Document(page_content="FAMILY TIP: The Nairobi National Park has a hidden 'Junior Ranger' program for kids aged 5-10."),
    Document(page_content="M-PESA DISCOUNT: Paying via M-Pesa at Fort Jesus gives you a free guided tour.")
]

vectorstore = InMemoryVectorStore.from_documents(
    documents=private_guide_content,
    embedding=embeddings
)

@tool
def check_private_travel_guide(query: str) -> str:
    """Consult the internal Magical Kenya private guide for secret discounts and family programs."""
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content


@tool
def get_weather(city: str) -> str:
    """Gets current weather for a city."""
    url  = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    data = requests.get(url).json()
    if "main" not in data:
        return f"Could not retrieve weather for '{city}'. Check the city name."
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"It is currently {temp}°C and {desc} in {city}."

@tool
def search_web(query: str) -> str:
    """Searches the web for local info and events."""
    if not SERPAPI_API_KEY:
        return "Search API key missing."
    resp = requests.get(
        "https://serpapi.com/search.json",
        params={"q": query, "api_key": SERPAPI_API_KEY}
    ).json()
    if "organic_results" not in resp:
        return "No search results found."
    return resp["organic_results"][0]["snippet"]


def build_travel_agent(model, tools_list):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional Kenya Travel Agent. Always use tools to verify weather or flights."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent_logic = create_tool_calling_agent(model, tools_list, prompt)
    return AgentExecutor(agent=agent_logic, tools=tools_list, verbose=True)

tools = [get_weather, search_web, check_private_travel_guide, verify_travel_feasibility]
agent = build_travel_agent(model=llm, tools_list=tools)

class ChatMessage(BaseModel):
    role: str      # "human" or "ai"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []   # client sends previous turns

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Reconstruct history from the request
    chat_history = []
    for msg in request.history:
        if msg.role == "human":
            chat_history.append(HumanMessage(content=msg.content))
        else:
            chat_history.append(AIMessage(content=msg.content))

    response = agent.invoke({
        "input": request.message,
        "chat_history": chat_history
    })
    ans = response.get("output", "")

        
        if isinstance(ans, list):
            ans = " ".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in ans
            )
        
        return {"reply": ans}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(current_dir), "index.html")  # go up one level
    if not os.path.exists(path):
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    with open(path, "r") as f:
        return f.read()










