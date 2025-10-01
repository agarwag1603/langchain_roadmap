from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", streaming=False)

fullspeech="""
“I have a dream that one day down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of interposition and nullification – one day right there in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers.
I have a dream today.
I have a dream that one day every valley shall be exalted, and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight, and the glory of the Lord shall be revealed and all flesh shall see it together.
This is our hope. This is the faith that I go back to the South with. With this faith we will be able to hew out of the mountain of despair a stone of hope. With this faith we will be able to transform the jangling discords of our nation into a beautiful symphony of brotherhood. With this faith we will be able to work together, to pray together, to struggle together, to go to jail together, to stand up for freedom together, knowing that we will be free one day.
This will be the day, this will be the day when all of God’s children will be able to sing with new meaning “My country ’tis of thee, sweet land of liberty, of thee I sing. Land where my father’s died, land of the Pilgrim’s pride, from every mountainside, let freedom ring!”
"""

generic_template = """
Write a summary of the following speech:
Speech: {speech}
/n/n
Also, translate the speech to the language: {language}
"""

prompt=  PromptTemplate(
    input_variables=['speech','language'],
    template=generic_template
)

str_output=StrOutputParser()

#This helps in complete prompt
#complete_prompt=prompt.format(speech=fullspeech,language="Hindi")


chain = prompt | gpt_llm | str_output
response=chain.invoke({"speech":fullspeech, "language":"French"})
print(response)