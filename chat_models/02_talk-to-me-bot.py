from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

# Initialize the model
llm = ChatOllama(model="deepseek-r1:1.5b")

# Instruct the model to always respond in JSON format
messages = [
    SystemMessage(content="You are a math assistant. Always respond in strict JSON format. Your response should have keys: 'question', 'answer', and 'explanation'."),
    HumanMessage(content="What is 81 divided by 9?"),
]

response = llm.invoke(messages)

# Output the raw response and try to parse it as JSON
print("Raw response:")
print(response.content)

# Try parsing the response to ensure it's valid JSON
try:
    data = json.loads(response.content)
    print("\nParsed JSON:")
    print(data)
except json.JSONDecodeError:
    print("\nThe response is not valid JSON.")
