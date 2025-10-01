from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

langsmith_doc=WebBaseLoader(web_path="https://docs.langchain.com/langsmith/manage-prompts")
langsmith_documents=langsmith_doc.load()

split_document=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
splitted_documents=split_document.split_documents(langsmith_documents)

openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

vectordbchroma=Chroma.from_documents(splitted_documents,openai_embedding)

chromaretriever=vectordbchroma.as_retriever()

system_prompt= """
You are an assistant for questioning the answer.
You should use the below context to answer the questions.
If you do not know the answer just be truthful and say you do not know.
Keep the answers very concise and do not give vague responses.
\n\n
{context}
"""

contextualize_q_system_prompt=(
    "Given a chat history and latest user questions"
    "which might reference context in the chat history,"
    "Just formulate the standalone question which can be understood without chat history" 
    "DO NOT answer the question, just reformulate"
)

                

contextualize_q_prompt=ChatPromptTemplate.from_messages([
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])


retriver_history_chain = create_history_aware_retriever(gpt_llm,chromaretriever,contextualize_q_prompt)

qa_prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

document_chain=create_stuff_documents_chain(gpt_llm,qa_prompt)

rag_chain = create_retrieval_chain(retriver_history_chain,document_chain)

chat_history=[]
question="What is commit tags?"
response=rag_chain.invoke({"input":question,"chat_history":chat_history})
print(response)

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=response['answer'])
    ]
)

print(60*"=")
question2="Tell me more about it?"
response2=rag_chain.invoke({"input":question2,"chat_history":chat_history})
print(response2)



