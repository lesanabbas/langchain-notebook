from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

llm = ChatOllama(model="deepseek-r1:1.5b")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}"),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}"),
    ]
)

netural_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escape_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human","Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

branches = RunnableBranch(
    (lambda x: "positive" in x, positive_feedback_template | llm | StrOutputParser()),
    (lambda x: "negative" in x, negative_feedback_template | llm | StrOutputParser()),
    (lambda x: "neutral" in x, netural_feedback_template | llm | StrOutputParser()),
    escape_feedback_template | llm | StrOutputParser()
)

classification_chain = classification_template | llm | StrOutputParser()

chain = classification_chain | branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "The product is okay. It works as expected but nothing exceptional."
result = chain.invoke({"feedback": review})
print(f"Result:\n{result}")