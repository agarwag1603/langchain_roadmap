##StructuredOutputParser can help you to have better responses based on your predefined field schema which is not possible with JsonOutputParser
##StructuredOutputParser can have enforced schema but can't have data validation

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
load_dotenv()

schema = [
    ResponseSchema(name="Joke_1",description="Joke with weak comedy"),
    ResponseSchema(name="Joke_2",description="Joke with mild comedy with analogy"),
    ResponseSchema(name="Joke_3",description="Joke with wild comedy with real life example")
]

parser = StructuredOutputParser.from_response_schemas(schema)

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

Prompt_Template = PromptTemplate(
    template="You need to tell me joke about \n \n  {topic} \n \n {format}.",
    input_variables=["topic"],
    partial_variables={"format":parser.get_format_instructions()}
)

chain = Prompt_Template | gpt_llm | parser

response=chain.invoke({"topic":"Cats"})
print(response)