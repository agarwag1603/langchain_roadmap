#Used for loading the text file into documents
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader= TextLoader("sample.txt")
text_documents=loader.load()
print(text_documents)
print(50*"*")
splitted_doc=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
final_document=splitted_doc.split_documents(text_documents)
for i in final_document:
    print(i)
