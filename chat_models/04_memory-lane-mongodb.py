import os
from dotenv import load_dotenv
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM

load_dotenv()

# MongoDB Atlas URI and config
MONGO_URI = os.getenv("MONGO_URI")
SESSION_ID = "user_1234"

# Setup chat history
history = MongoDBChatMessageHistory(
    connection_string=MONGO_URI,
    session_id=SESSION_ID,
    database_name="langchain_db",
    collection_name="chat_history",
)

# Initialize Ollama LLM
llm = OllamaLLM(model="deepseek-r1:1.5b")

# Chat loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    history.add_message(HumanMessage(content=query))
    response = llm.invoke(history.messages)

    print("AI:", response)
    history.add_message(AIMessage(content=response))
