##This program deals with enforcing the llm output in certain way. Using annotated to get the list of strings, literals for hard coded sentiments

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional,Literal
from pydantic import BaseModel,Field
load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Product(BaseModel):
    key_points: list[str]=  Field(description="Write me all the key pointers related to the product")
    price : int = Field(description="Find price for the product")
    sentiment: Literal["Very Good","Good","Bad","Very Bad"] = Field(description="Find me the sentiments of the product")
    product_summary : str = Field(description="Write me a product summary")
    pros : Optional[list[str]] =  Field (description="Find the pros for the product")
    #cons :Optional[list[str]] =  Field ("Find the cons for the product")
    cons :list[str] =  Field (default=None, description="Find the cons for the product")

structured_llm = gpt_llm.with_structured_output(Product)

product_review="""ChatGPT has been a revolutionary technology built by OpenAI. 
It has many models that supports text generation, image generation, and voice generation.
It has been transformative in the field of customer support, medicine, sports and many others.
I think it does the job well, but also hallucinate time to time. It needs to be trained on very large data and fine tuned to have a better inference.
It's around $20 a month for the model usage.

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
