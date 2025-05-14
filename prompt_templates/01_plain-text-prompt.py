from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate


llm = ChatOllama(model="deepseek-r1:1.5b")


# Create ChatPromptTemplate using a template string

# template = "Tell me a joke about {topic}."
# result = llm.invoke(template.format(topic="cats"))
# print(f"AI: {result.content}")


# Create a Prompt Multiple Placeholders

# template = """You are a helpful assistant. Human: Tell me a {objective} story of a {animal}. Assistant:"""
# result = llm.invoke(template.format(objective="funny", animal="cat"))
# print(f"AI: {result.content}")


# Prompt with System and HumanMessage (using tuples)

# messages = [
#     ('system', "You are a comedian who can tell jokes about {topic}."),
#     ('human', "Tell me joke {joke_number} jokes."),
# ]
# template = ChatPromptTemplate.from_messages(messages)
# prompt = template.format_messages(topic="cats", joke_number=1)
# result = llm.invoke(prompt)
# print(f"AI: {result.content}")


# Extra Information about above example:

# messages = [
#     ('system', 'you are the comedian who can tell jokes about {topic}.'),
#     HumanMessage(content="Tell me joke 3 jokes."),
# ]
# template = ChatPromptTemplate.from_messages(messages)
# prompt = template.format_messages(topic="lawyers")
# result = llm.invoke(prompt)
# print(f"AI: {result.content}")


# Below example will not work to show that HumanMessage needs proper formatting to work with variables.

messages = [
    ('system', 'you are the comedian who can tell jokes about {topic}.'),
    HumanMessage(content="Tell me {joke_count} jokes."),
]
template = ChatPromptTemplate.from_messages(messages)
prompt = template.format_messages(topic="lawyers")
print(f"Prompt: {prompt}")
result = llm.invoke(prompt)
print(f"AI: {result.content}")
