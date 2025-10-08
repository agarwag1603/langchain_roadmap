##RePhraseQuery is a simple retriever that applies an LLM between the user input and the query passed by the retriever.

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain.retrievers import RePhraseQueryRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import logging


load_dotenv()

logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)

loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=384)

vectorstore = Chroma.from_documents(all_splits, openai_embedding)

gpt_llm = ChatOpenAI(model="gpt-4o-mini")

prompt_template= ChatPromptTemplate.from_messages (
    [
        ("system","You are an AI assistant that helps in expanding the query if necessary"),
        ("human","Answer the query based on subject: \n\n {subject}")

    ]
)

retriever_from_llm = RePhraseQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=gpt_llm
)

chains = retriever_from_llm | {"subject":RunnablePassthrough()} |  prompt_template | gpt_llm | StrOutputParser()

docs = chains.invoke(
    "Langchain has been a real good deal. Have been focus on integrating all the AI LLMs"
)

print(docs)

#Output:

# INFO:langchain.retrievers.re_phraser:Re-phrased question: To convert the user's query into a more concise and relevant form for a vectorstore, we can focus on the key concepts related to Langchain and AI LLM integration. Here’s the refined query:

# "Langchain AI LLM integration"
# The subject matter appears to focus on LangChain, a framework for developing applications that utilize large language models (LLMs). Based on the information provided, the following queries can be explored or expanded:

# 1. **What is LangChain?** 
#    - An overview of the framework, its purpose, and how it integrates with large language models.

# 2. **How does LangChain simplify the application lifecycle?**
#    - Discussion on the different stages of the LLM application lifecycle that LangChain simplifies.

# 3. **What are the capabilities of LangChain?**
#    - Details on functionalities such as handling semantic similarity, streaming responses, validation and processing of SQL queries, and tools for custom function invocation.

# 4. **How can I integrate LangChain with various data sources?**
#    - Insights on dealing with databases, CSVs, and using retrieval mechanisms in chatbots.

# 5. **What are the migration paths within LangChain?**
#    - Information on how users can migrate from older versions of chains and integrate new features.

# 6. **What resources and tools are available in the LangChain ecosystem?**
#    - A look into supporting resources like LangSmith and LangGraph.

# 7. **Installation guide for LangChain.**
#    - A step-by-step on how to install LangChain packages and dependencies.

# 8. **Using examples in LangChain for enhanced query analysis.**
#    - Exploration of how few-shot examples and demonstrations can improve the performance and reliability of responses.

# 9. **Advanced functionalities and use cases of LangChain.**
#    - Real-world applications and scenarios where LangChain can be effectively utilized.

# Feel free to choose any of the topics above for further exploration, or let me know if there's a specific area of interest related to LangChain you would like to delve deeper into!