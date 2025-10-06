#It's a type dict with_structured_output. Even when you do not give a chat_template, a template is created automatically to enforce the output.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Product(TypedDict):
    name: str
    sentiment: str
    product_summary : str


structured_model = gpt_llm.with_structured_output(Product)

topic= """
Gaming console is amazing for XBOX. I really love every bit of it. Pixel is so high and well cureated.
"""

response=structured_model.invoke(topic)
print(response)
print(response['name'])
print(response['product_summary'])
print(response['sentiment'])

################################################################################################################

#It's a type dict with_structured_output with a chatprompt template to enforce an output.
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
chain =  structured_model

topic= """
ChatGPT at times Hallucinate but does the job for me. Chat conversations are well intact although I would like it to be more efficient.
"""

response=chain.invoke(topic)
print(response)
print(response['name'])
print(response['product_summary'])
print(response['sentiment'])

