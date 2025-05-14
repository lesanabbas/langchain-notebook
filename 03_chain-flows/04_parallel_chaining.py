from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableMap

# Initialize model
llm = ChatOllama(model="deepseek-r1:1.5b")
parser = StrOutputParser()

# Step 1: Extract product features
feature_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)
feature_chain = feature_prompt | llm | parser


# Step 2: Define pros analysis chain
def analyze_pros():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return prompt | llm | parser


# Step 3: Define cons analysis chain
def analyze_cons():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return prompt | llm | parser


# Step 4: Combine the analyses
def combine_analysis(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Step 5: Create full chain
analyze_pros_chain = RunnableLambda(lambda x: {"features": x}) | analyze_pros()
analyze_cons_chain = RunnableLambda(lambda x: {"features": x}) | analyze_cons()

# Use RunnableParallel to run pros and cons together
parallel_chain = RunnableParallel(
    pros=analyze_pros_chain,
    cons=analyze_cons_chain,
)

# Full pipeline
full_chain = (
    feature_chain
    | parallel_chain
    | RunnableLambda(lambda x: combine_analysis(x["pros"], x["cons"]))
)

# Run the chain
result = full_chain.invoke({"product_name": "iPhone 14"})
print(f"Result:\n{result}")
