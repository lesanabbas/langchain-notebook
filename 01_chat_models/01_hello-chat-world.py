from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="deepseek-r1:1.5b")

result = llm.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)

# If result is a string (most likely), just print it
print("Content only:")
print(str(result))
