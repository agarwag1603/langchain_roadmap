##This program deals with enforcing the llm output in certain way. Using annotated to get the list of strings, literals for hard coded sentiments

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Product(TypedDict):
    key_points: Annotated[list[str],"Write me all the key pointers related to the product"]
    sentiment: Annotated[Literal["Very Good","Good","Bad","Very Bad"],"Find me the sentiments of the product"]
    product_summary : Annotated[str,"Write me a product summary"]
    pros : Annotated[list[str],"Find the pros for the product"]
    cons : Annotated[list[str],"Find the cons for the product"]

structured_llm = gpt_llm.with_structured_output(Product)

product_review="""ChatGPT has been a revolutionary technology built by OpenAI. 
It has many models that supports text generation, image generation, and voice generation.
It has been transformative in the field of customer support, medicine, sports and many others.
I think it does the job well, but also hallucinate time to time. It needs to be trained on very large data and fine tuned to have a better inference.

Pros: 
-Generate Code very quick
-Debug the issues very quick
-Generate poems

Cons:
-Hallucinates and makes data up
-Doesn't have latest information
-Can forge someone's identity
"""

response= structured_llm.invoke(product_review)
print(response)