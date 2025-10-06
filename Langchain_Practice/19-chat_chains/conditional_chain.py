from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableLambda, RunnableBranch

load_dotenv()

gpt_4_llm = ChatOpenAI(model="gpt-4o-mini")

class FactChecker(BaseModel):
    fact: Literal["true", "false"] = Field(default=None, description="Get the fact value as true or false")

pydantic_parser= PydanticOutputParser(pydantic_object=FactChecker)

Chat_Prompt_template= ChatPromptTemplate.from_messages(
    [
        ("system","Your role is to provide the factual information in true or false"),
        ("human","Please help in validate the truthness behind the statement for the topic: \n {topic} \n {format}")
    ]
)

chat_prompt = Chat_Prompt_template.partial(format=pydantic_parser.get_format_instructions())

fact_chain = chat_prompt | gpt_4_llm | pydantic_parser

Chat_Prompt_truth= ChatPromptTemplate.from_messages(
    [
        ("system","Your role is to provide the fun fact"),
        ("human","Please provide a fun fact about the topic: \n {topic}")
    ]
)

Chat_Prompt_false= ChatPromptTemplate.from_messages(
    [
        ("system","Your role is to correct the false statement"),
        ("human","Please correct the fact if it's false: \n {topic}")
    ]
)

branch_chain = RunnableBranch (
    (lambda x:x.fact == "true", Chat_Prompt_truth | gpt_4_llm | StrOutputParser()),
    (lambda x:x.fact == "false", Chat_Prompt_false | gpt_4_llm | StrOutputParser()),
    RunnableLambda(lambda x:"Could not find the fact")
)

final_chain = fact_chain | branch_chain

result=final_chain.invoke({"topic":"Earth has only 2 moon."})

print(result)


