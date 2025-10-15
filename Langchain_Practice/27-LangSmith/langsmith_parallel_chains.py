from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough,RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT']="Parallel RunnablePassthough Sequential"

gpt_llm = ChatOpenAI(model="gpt-4o-mini")

para_prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant who only rights 3 paragaraphs about the user query."),
        ("human","Kindly, help in generating the paragraphs for the the user query: \n {query}")
    ]
)

blogs_prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant who only rights short blogs related to the user query."),
        ("human","Kindly, help in generating the blog for the the user query: \n {query}")
    ]
)

summary_prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system","Based on the user input, your role is to provide a summary by combining the responses."),
        ("human","Please help in generating summary but make sure you you combine blogs and paragaraphs: {para_response} \n \n {blog_response}")
    ]
)

keypoint_prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system","Based on the user input, your role is to provide a key notes in bullet points"),
        ("human","Please help in generating keypoints for the summary: {keypoints}")
    ]
)


parallel_chain=RunnableParallel(
    {
    "para_response":para_prompt_template | gpt_llm | StrOutputParser(),
    "blog_response":blogs_prompt_template | gpt_llm | StrOutputParser()
    }
)

keypoints_runnable = RunnableParallel(
    {
        "summary":RunnablePassthrough(),
        "keypoints":RunnableSequence(keypoint_prompt_template, gpt_llm, StrOutputParser())
    }
)

config = {
    "run_name":"Parallel-RunnablePassthough-Sequential"
}

summary_chain = parallel_chain | summary_prompt_template | gpt_llm | StrOutputParser ()

final_chain= summary_chain | keypoints_runnable

print(final_chain.invoke({"query":"What is SVM?"},config=config))