from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json

# Initialize ChatOllama with deepseek model
llm = ChatOllama(model="deepseek-r1:1.5b")

# Start the chat history
chat_history = [
    SystemMessage(
        content="You are a helpful AI assistant. Always respond strictly in JSON format with keys: 'question', 'answer', and 'explanation'."
    )
]

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    # Append user message
    chat_history.append(HumanMessage(content=query))

    # Invoke the model
    result = llm.invoke(chat_history)

    # Parse AI response
    try:
        parsed = json.loads(result.content)
        print("AI:")
        print(json.dumps(parsed, indent=2))
        chat_history.append(AIMessage(content=result.content))  # save clean JSON
    except json.JSONDecodeError:
        print("AI returned invalid JSON:")
        print(result.content)
        chat_history.append(AIMessage(content=result.content))

# Print entire chat history (optional)
print("\nChat History:")
for msg in chat_history:
    role = msg.type
    print(f"{role.upper()}: {msg.content}")
