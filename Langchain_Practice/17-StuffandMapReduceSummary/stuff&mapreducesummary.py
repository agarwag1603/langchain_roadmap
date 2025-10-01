from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

pdf_loader=PyPDFLoader("Langchain_Practice/17-StuffandMapReduceSummary/Speeches.pdf")
pdf_documents=pdf_loader.load()
print(pdf_documents)

splitted_doc=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap =50).split_documents(pdf_documents)
print(splitted_doc)

gpt_llm = ChatOpenAI(model="gpt-4o-mini", streaming=False)

template="""
write a concise summary of this document for the following speech:
\n\n
{text}
"""

print(template.format(text=splitted_doc))

prompt= PromptTemplate(input_variables=["text"],template=template)

chain=load_summarize_chain(llm=gpt_llm, chain_type="stuff", prompt=prompt, verbose=True)
print(chain.run(splitted_doc))