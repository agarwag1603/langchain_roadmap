## Retrieval is done based on KNN - Doesn't perform a good search.
from langchain_community.retrievers import KNNRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini")

docs = [
    Document(
        page_content="Glaciers are melting because of global warming. Michael Jackson has been a real good performer.",
        metadata={"topic": "Environment", "type": "Climate", "id": 1}
    ),
    Document(
        page_content="Java is widely used for enterprise applications and runs on the Java Virtual Machine (JVM).",
        metadata={"topic": "Java", "type": "programming", "id": 2}
    ),
    Document(
        page_content="Antartica has not been habitable anymore due to ice turning to water. I love skates.",
        metadata={"topic": "Planet", "type": "Climate", "id": 3}
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

vectorstore = KNNRetriever.from_documents(docs, openai_embedding)

results = vectorstore.invoke("Glacier is melting")
for r in results:
    print(r.metadata, "→", r.page_content)


#Output
# {'topic': 'Environment', 'type': 'Climate', 'id': 1} → Glaciers are melting because of global warming. Michael Jackson has been a real good performer.
# {'topic': 'Planet', 'type': 'Climate', 'id': 3} → Antartica has not been habitable anymore due to ice turning to water. I love skates.
# {'topic': 'RAG', 'type': 'AI Concept', 'id': 6} → Retrieval-Augmented Generation (RAG) combines external knowledge retrieval with generative models to improve factual accuracy.
# {'topic': 'OpenAI', 'type': 'AI Company', 'id': 4} → OpenAI develops models such as GPT-4 and GPT-4o-mini, which are used for text generation and reasoning tasks.