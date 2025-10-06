##PydanticOutputParser helps in enforcing the schema
## helps in data validation too

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class UserInformation(BaseModel):
    name: str  = Field(description="Name of the personality")
    age: int = Field(description="Age of the personality")
    birthplace: str = Field(description="Birthplace of the personality")

parser=PydanticOutputParser(pydantic_object=UserInformation)

prompt_template = PromptTemplate(
    template=("Give the fictional name, age, birthplace for following race: {race} \n\n in format {format}"),
    input_variables=["race"],
    partial_variables={"format":parser.get_format_instructions()}
)

# Build chain properly
chain = prompt_template | gpt_llm | parser

# Run
structured_response = chain.invoke({"race": "Indian"})

print(structured_response)