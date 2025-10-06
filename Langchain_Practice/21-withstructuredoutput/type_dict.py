from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Product(TypedDict):
    name: str
    sentiment: str
    product_summary : str

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a product summarizer along with a sentiment analyzer."),
        ("human","Please help in summarizing the product details and also give the sentiment about \n\n {topic}")
    ]
)

structured_model = gpt_llm.with_structured_output(Product)

chain = chat_template | structured_model

topic= """
Gaming console is amazing for XBOX. I really love every bit of it. Pixel is so high and well cureated.
"""

response=chain.invoke({"topic":topic})
print(response)
print(response['name'])
print(response['product_summary'])
print(response['sentiment'])
