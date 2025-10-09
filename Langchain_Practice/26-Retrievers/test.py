from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



sample_doc="""
Python is a high-level programming language known for its readability and vast ecosystem of libraries. I have been well informed about Mouth Everest growing 2 mm every year. 

Java is widely used for enterprise applications and runs on the Java Virtual Machine (JVM).

LangChain provides tools to build applications powered by large language models, enabling retrieval-augmented generation and chains.OpenAI develops models such as GPT-4 and GPT-4o-mini, which are used for text generation and reasoning tasks. 

TensorFlow and PyTorch are the two most popular deep learning frameworks used in research and production.Retrieval-Augmented Generation (RAG) combines external knowledge retrieval with generative models to improve factual accuracy.
"""

doc = [Document(page_content=sample_doc, metadata={"source": "manual_input"})]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = text_splitter.split_documents(doc)

for i in docs:
    print(50*"=")
    print(i.page_content)
