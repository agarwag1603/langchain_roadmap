# Wikipedia retriver works in retriving the data from wikipedia
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnablePassthrough  
from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv

load_dotenv()

retriever=WikipediaRetriever(top_k_results=2,lang="en",)

prompt = ChatPromptTemplate.from_template(  
"""Answer the question based only on the context provided.  
Context: {context}  
Question: {question}"""  
)  
gpt_llm = ChatOpenAI(model="gpt-4o-mini")  
def format_docs(docs):  
    return "\n\n".join(doc.page_content for doc in docs)  
chain = (  
{"question": RunnablePassthrough(),"context": retriever | format_docs }  
| prompt  
| gpt_llm  
| StrOutputParser()  
)  

print(chain.invoke(  
"who is the main character of the show 'The Office?'"  
))

# #Output
# The main character of the show 'The Office' is Michael Scott, portrayed by Steve Carell, who is the regional manager of Dunder Mifflin in Scranton, Pennsylvania.


