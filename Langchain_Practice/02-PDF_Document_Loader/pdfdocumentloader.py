from langchain_community.document_loaders import PyPDFLoader

pdf_loader= PyPDFLoader("Langchain_Practice/02-PDF_Document_Loader/Speeches.pdf")
pdf_documents=pdf_loader.load()
print(pdf_documents)