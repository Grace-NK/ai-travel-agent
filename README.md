# Magical Kenya AI Travel Agent

For this project, I was required to build a sophisticated, production-ready AI Travel Assistant capable of providing real-time travel advice, verifying budget feasibility, and accessing private local knowledge about Kenya. The goal was to move beyond a simple chatbot and create an "Agentic System" that can reason and use external tools to solve user queries.

🏗 The Process
My development followed a structured 5-stage sprint:

Logic & Feasibility: Developed a specialized "Logic Engine" using Pydantic schemas to force the LLM to perform structured budget analysis.

RAG Integration (Retrieval-Augmented Generation): Built a local vector database (ChromaDB) to store and retrieve "secret" local tips, such as M-Pesa discounts and hidden family programs in Nairobi.

Tool Orchestration: Integrated multiple external APIs (Amadeus for flights, OpenWeather for climate, and SerpAPI for live web search) to give the agent "eyes and ears."

Memory Management: Implemented a conversation buffer using LangChain's MessagesPlaceholder so the agent remembers user names and previous context.

Cloud Deployment: Transitioned the local Jupyter environment into a FastAPI backend optimized for serverless deployment on Vercel.

🛠 Tools & Technologies
LLM: Google Gemini 1.5 Flash (chosen for its speed and high-accuracy tool calling).

Framework: LangChain (AgentExecutor, Tool Calling Agent).

Database: ChromaDB (Vector Store) with HuggingFace Embeddings.

APIs: Amadeus (Travel), OpenWeather (Live Weather), SerpAPI (Google Search).

Backend: FastAPI & Uvicorn.

Deployment: Vercel (CI/CD via GitHub).

🚀 Key Features
Budget Guardrails: The agent won't just say "yes" to a trip; it analyzes if the user's budget is realistic for the destination.

Private Knowledge: Access to exclusive "Magical Kenya" tips not found on the general web.

Live Data: Real-time flight pricing and weather updates.

Contextual Memory: Ability to maintain a natural conversation over multiple turns.

📈 Outcome
The final result is a live, scalable API that demonstrates competency in AI Orchestration, API integration, and secure cloud deployment. It successfully handles the "hallucination" problem by grounding its answers in real-time data and private documents.
