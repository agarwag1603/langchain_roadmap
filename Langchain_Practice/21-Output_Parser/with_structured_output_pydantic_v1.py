from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class UserInformation(BaseModel):
    name: str  = Field(description="Name of the personality")
    age: int = Field(description="Age of the personality")
    birthplace: str = Field(description="Birthplace of the personality")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot which only fetches information from Wikipedia. "),
    ("human", "Provide me the information about this personality: {personality}")
])

# Wrap LLM with structured output
structured_gpt_llm = gpt_llm.with_structured_output(UserInformation)

# Build chain properly
chain = chat_prompt | structured_gpt_llm

# Run
structured_response = chain.invoke({"personality": "Roger Federer"})

print(structured_response)