##All types of search based retrievers -  similarity search/Maximal Marginal Relevance/Contextual Compression Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI  

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini")

docs = [
    Document(
        page_content="""Python is a high-level programming language known for its readability and vast ecosystem of libraries. 
        I have been well informed about Mouth Everest growing 2 mm every year.""",
        metadata={"topic": "Python", "type": "programming", "id": 1}
    ),
    Document(
        page_content="Java is widely used for enterprise applications and runs on the Java Virtual Machine (JVM).",
        metadata={"topic": "Java", "type": "programming", "id": 2}
    ),
    Document(
        page_content="LangChain provides tools to build applications powered by large language models, enabling retrieval-augmented generation and chains.",
        metadata={"topic": "LangChain", "type": "AI Framework", "id": 3}
    ),
    Document(
        page_content="OpenAI develops models such as GPT-4 and GPT-4o-mini, which are used for text generation and reasoning tasks.",
        metadata={"topic": "OpenAI", "type": "AI Company", "id": 4}
    ),
    Document(
        page_content="TensorFlow and PyTorch are the two most popular deep learning frameworks used in research and production.",
        metadata={"topic": "Deep Learning", "type": "framework", "id": 5}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines external knowledge retrieval with generative models to improve factual accuracy.",
        metadata={"topic": "RAG", "type": "AI Concept", "id": 6}
    ),
]

openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=384)
vectorstore= Chroma.from_documents(docs, openai_embedding)

print(50*"=")
print("Output with Similarity search")
print(50*"=")
retriever_similarity = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
results = retriever_similarity.invoke("What is LangChain?")
for r in results:
    print(r.metadata, "→", r.page_content)
print(50*"=")
print("Output with lambda_mult:1")
print(50*"=")
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 1})
results = retriever_mmr.invoke("AI frameworks and tools for developers")
for r in results:
    print(r.metadata, "→", r.page_content)
print(50*"=")
print("Output with lambda_mult:0.5")
print(50*"=")
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5})
results = retriever_mmr.invoke("AI frameworks and tools for developers")
for r in results:
    print(r.metadata, "→", r.page_content)
print(50*"=")
print("Output with Contextual Compression Retriever")
print(50*"=")
retriever_ccr = vectorstore.as_retriever(search_kwargs={"k": 3})

compressor = LLMChainExtractor.from_llm(gpt_llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever_ccr
)

compressed_docs = compression_retriever.invoke(
    "I love Mountains and trekking."
)

print(compressed_docs)


# ==================================================
# Output with Similarity search
# ==================================================
# {'id': 3, 'topic': 'LangChain', 'type': 'AI Framework'} → LangChain provides tools to build applications powered by large language models, enabling retrieval-augmented generation and chains.
# {'topic': 'RAG', 'type': 'AI Concept', 'id': 6} → Retrieval-Augmented Generation (RAG) combines external knowledge retrieval with generative models to improve factual accuracy.
# {'type': 'AI Company', 'topic': 'OpenAI', 'id': 4} → OpenAI develops models such as GPT-4 and GPT-4o-mini, which are used for text generation and reasoning tasks.
# ==================================================
# Output with lambda_mult:1
# ==================================================
# {'topic': 'OpenAI', 'id': 4, 'type': 'AI Company'} → OpenAI develops models such as GPT-4 and GPT-4o-mini, which are used for text generation and reasoning tasks.
# {'id': 3, 'type': 'AI Framework', 'topic': 'LangChain'} → LangChain provides tools to build applications powered by large language models, enabling retrieval-augmented generation and chains.
# {'id': 2, 'topic': 'Java', 'type': 'programming'} → Java is widely used for enterprise applications and runs on the Java Virtual Machine (JVM).
# ==================================================
# Output with lambda_mult:0.5
# ==================================================
# {'type': 'AI Company', 'topic': 'OpenAI', 'id': 4} → OpenAI develops models such as GPT-4 and GPT-4o-mini, which are used for text generation and reasoning tasks.
# {'id': 2, 'type': 'programming', 'topic': 'Java'} → Java is widely used for enterprise applications and runs on the Java Virtual Machine (JVM).
# {'type': 'programming', 'id': 1, 'topic': 'Python'} → Python is a high-level programming language known for its readability and vast ecosystem of libraries. 
#         I have been well informed about Mouth Everest growing 2 mm every year.
# ==================================================
# Output with Contextual Compression Retriever
# ==================================================
# [Document(metadata={'topic': 'Python', 'id': 1, 'type': 'programming'}, page_content='I have been well informed about Mouth Everest growing 2 mm every year.')]



