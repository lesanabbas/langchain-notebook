from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence


llm = ChatOllama(model="deepseek-r1:1.5b")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a comedian who can tell jokes about {topic}"),
        ('human', "Tell me {joke_count} jokes."),
    ]
)

# Defining additional processing steps using RunnableLambda
uppercase = RunnableLambda(lambda x: x.lower())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n {x}")

chain = prompt_template | llm | StrOutputParser() | uppercase | count_words

result = chain.invoke({"topic": "cats", "joke_count": 2})
print(f"Result: {result}")