from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence

llm = ChatOllama(model="deepseek-r1:1.5b")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a comedian who can tell jokes about {topic}"),
        ('human', "Tell me {joke_count} jokes."),
    ]
)

# Create individual runnables (step in the chain)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
format_llm = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[format_llm], last=parse_output)
response = chain.invoke({"topic": "cats", "joke_count": 2})
print(f"Result: {response}")