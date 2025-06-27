from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")    


def get_llm():
    # You can add more config or caching here if needed
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
