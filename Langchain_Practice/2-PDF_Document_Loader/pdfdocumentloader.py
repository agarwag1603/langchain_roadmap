from langchain_community.document_loaders import PyPDFLoader

pdf_loader= PyPDFLoader("Itinerary_aus_nz.pdf")
pdf_documents=pdf_loader.load()
print(pdf_documents)