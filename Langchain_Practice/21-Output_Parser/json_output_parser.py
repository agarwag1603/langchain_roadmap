##The json strcutured output can be used for the open source llm which do not work well with with_structured_output
##But biggest drawback of JSON output parser is it doesn't enforce schema and data validations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

parser = JsonOutputParser()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

Prompt_Template = PromptTemplate(
    template="Find me the value of name, age, city \n in {format} for the {topic}",
    input_variables=["topic"],
    partial_variables={"format":parser.get_format_instructions()}
)

topic = "Gaurav Agarwal was born in India, Hazaribag on 16th Jan 1992"

chain = Prompt_Template | gpt_llm | parser

print(chain.invoke())

