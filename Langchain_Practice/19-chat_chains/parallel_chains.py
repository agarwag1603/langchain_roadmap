from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

gpt_5_llm = ChatOpenAI(model="gpt-5-mini")
gpt_4_llm = ChatOpenAI(model="gpt-4o-mini")


chat_prompt_notemaker= ChatPromptTemplate.from_messages([
    ("system","You're a langchain tutor, you are most of the times busy making notes to be distributed to kids."),
    ("human","Make 5 notes related to the subject : \n\n {subject}")
])

chat_prompt_quizmaker= ChatPromptTemplate.from_messages([
    ("system","You're a langchain quiz maker and assist students with quiz that has question and answers."),
    ("human","Help in curating 3 quiz questions from the subject: \n\n {subject}")
])

chat_prompt_merged=ChatPromptTemplate.from_messages([
    ("system","You role is to merge the input from notes and quiz"),
    ("human","Help in merging the notes and quiz but keep them sepatated please: \n\n Notes: {notes} \n\n Quiz: {quiz}")
])


runnable_chain = RunnableParallel({
    "notes": chat_prompt_notemaker | gpt_5_llm | StrOutputParser(),
    "quiz": chat_prompt_quizmaker | gpt_4_llm | StrOutputParser()
})

merged_chain = chat_prompt_merged | gpt_4_llm | StrOutputParser()

final_chain = runnable_chain | merged_chain

subject="""
What are the core components of LangChain?
Using LangChain, software teams can build context-aware language model systems with the following modules. 

LLM interface
LangChain provides APIs with which developers can connect and query LLMs from their code. Developers can interface with public and proprietary models like GPT, Bard, and PaLM with LangChain by making simple API calls instead of writing complex code.

Prompt templates
Prompt templates are pre-built structures developers use to consistently and precisely format queries for AI models. Developers can create a prompt template for chatbot applications, few-shot learning, or deliver specific instructions to the language models. Moreover, they can reuse the templates across different applications and language models. 

Agents
Developers use tools and libraries that LangChain provides to compose and customize existing chains for complex applications. An agent is a special chain that prompts the language model to decide the best sequence in response to a query. When using an agent, developers provide the user's input, available tools, and possible intermediate steps to achieve the desired results. Then, the language model returns a viable sequence of actions the application can take.  

Retrieval modules
LangChain enables the architecting of RAG systems with numerous tools to transform, store, search, and retrieve information that refine language model responses. Developers can create semantic representations of information with word embeddings and store them in local or cloud vector databases. 

Memory
Some conversational language model applications refine their responses with information recalled from past interactions. LangChain allows developers to include memory capabilities in their systems. It supports:

Simple memory systems that recall the most recent conversations. 
Complex memory structures that analyze historical messages to return the most relevant results. 
Callbacks
Callbacks are codes that developers place in their applications to log, monitor, and stream specific events in LangChain operations. For example, developers can track when a chain was first called and errors encountered with callbacks. 
"""

print(final_chain.invoke({"subject":subject}))




    