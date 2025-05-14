from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOllama(model="deepseek-r1:1.5b")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a comedian who can tell jokes about {topic}"),
        ('human', "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain using LangChain Expressions Language (LCEL)

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"topic": "cats", "joke_count": 2})
print(f"Result: {result}")