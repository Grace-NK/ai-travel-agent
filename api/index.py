import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from amadeus import Client, ResponseError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.utilities import SerpAPIWrapper

app = FastAPI()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
client = genai.Client(api_key=GOOGLE_API_KEY)

# Feasible logic

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
        config={'response_mime_type': 'application/json', 'response_schema': TravelPlan}
    )
    return response.text

private_guide_content = [
    Document(page_content="SECRET DEAL: Use code 'JAMBO2026' for 20% off at any Diani beach resort."),
    Document(page_content="FAMILY TIP: The Nairobi National Park has a hidden 'Junior Ranger' program for kids aged 5-10."),
    Document(page_content="M-PESA DISCOUNT: Paying via M-Pesa at Fort Jesus gives you a free guided tour.")
]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# This keeps the database in the RAM so Vercel doesn't crash trying to write to disk
vectorstore = Chroma.from_documents(
    documents=private_guide_content, 
    embedding=embeddings,
    persist_directory=None  
)

@tool
def check_private_travel_guide(query: str) -> str:
    """Consult the internal Magical Kenya private guide for secret discounts and family programs."""
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content

@tool
def get_weather(city: str) -> str:
    """Gets current weather for a city."""
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    data = requests.get(url).json()
    if "main" not in data:
        return f"Could not retrieve weather for '{city}'. Check the city name."
    temp = data["main"]["temp"]
    return f"It is currently {temp}°C in {city}."

@tool
def search_web(query: str) -> str:
    """Searches Google for local info and events."""
    search = SerpAPIWrapper()
    return search.run(query)

@tool
def flight_search(origin: str, destination: str, date: str) -> str:
    """Finds flights (Dates: YYYY-MM-DD, Codes: NBO, MBA)."""
    amadeus = Client(client_id=os.environ["AMADEUS_API_KEY"], 
                     client_secret=os.environ["AMADEUS_API_SECRET"])
    try:
        resp = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin, destinationLocationCode=destination, 
            departureDate=date, adults=1, max=1)
        price = resp.data[0]['price']['total']
        curr = resp.data[0]['price']['currency']
        return f"Found a flight for {price} {curr}."
    except Exception as e:
        return f"No flights found. ({e})"

def build_travel_agent(model, tools_list):
    """
    Constructs the agent's logic and executor.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional Kenya Travel Agent. Always use tools to verify weather or flights."),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}"),                             
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ])
 
    agent_logic = create_tool_calling_agent(model, tools_list, prompt)
    return AgentExecutor(agent=agent_logic, tools=tools_list, verbose=True)
    
tools = [get_weather, search_web, flight_search, check_private_travel_guide, verify_travel_feasibility]

agent = build_travel_agent(model=llm, tools_list=tools)

chat_history = []

def ask_agent(user_input):
    response = agent.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    ans = response["output"]
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=ans))
    
    return ans

print("Agent is ready for action!")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = agent.invoke({
        "input": request.message,
        "chat_history": chat_history
    })
    ans = response["output"]
    chat_history.append(HumanMessage(content=request.message))
    chat_history.append(AIMessage(content=ans))

    return {"reply": ans}





