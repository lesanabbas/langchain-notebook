from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

llm = ChatOllama(model="deepseek-r1:1.5b")

# Create ChatPromptTemplate using a template string

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.format_messages(topic="cats")

result = llm.invoke(prompt)
print(f"AI: {result.content}")


# Prompt with multiple placeholders 

template = """You are a helpful assistant. Human: Tell me a {adjective} short stroy about a {animal}. Assistant:"""
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.format_messages(adjective="funny", animal="cat")
result = llm.invoke(prompt)
print(f"AI: {result.content}")


# Prompt with System and HumanMessages (using tuples)

messages = [
    ('system', "You are a comedian who can tell jokes about {topic}"),
    ('human', "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.format_messages(topic="cats", joke_count=3)
result = llm.invoke(prompt)
print(f"AI: {result.content}")