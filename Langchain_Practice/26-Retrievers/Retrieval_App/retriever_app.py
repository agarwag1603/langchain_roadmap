##Retrievers type with streamlit app - Retrieval_Search_App.png is added to the project 
import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini")
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)

st.title("🔍 Search Type Retriever")

user_docs = st.text_area(
    "Enter your document data",
    height=200,
    placeholder="Paste a long paragraph, ticket, or document text here..."
)


# --- Submit Button ---
if st.button("Load Document"):
    if user_docs.strip():
        # Clear all previous session state to reset everything
        for key in list(st.session_state.keys()):
            if key != "user_docs":
                del st.session_state[key]
        
        doc = [Document(page_content=user_docs, metadata={"source": "manual_input"})]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        docs = text_splitter.split_documents(doc)

        # Delete existing vectorstore if it exists
        try:
            if "vectorstore" in st.session_state:
                st.session_state["vectorstore"].delete_collection()
        except:
            pass

        # Create new vectorstore with unique collection name
        vectorstore = Chroma.from_documents(
            docs, 
            openai_embedding,
            collection_name=f"session_{id(st.session_state)}"
        )
        
        st.session_state["vectorstore"] = vectorstore
        st.session_state["doc_loaded"] = True
        st.success("Document embedding completed successfully.")
        st.rerun()  # Force reload to reset UI fields
    else:
        st.warning("Please enter some text before loading.")

if st.session_state.get("doc_loaded", False):

    st.header("Retrieve from your documents")

    search_option = st.selectbox(
        "Select your retriever action",
        ("", "Similarity Search", "Maximal Marginal Relevance search", "Contextual Compression Retriever"),
        key="search_option"
    )

    if search_option:
        search_arguments = st.slider(
            "Select the number of documents to be searched",
            min_value=1,
            max_value=5,
            value=3,
            key="search_arguments"
        )

        user_query = st.text_input("Enter your query:", key="user_query")
        
        if st.button("Search"):
            if user_query.strip():
                vectorstore = st.session_state["vectorstore"]

                with st.spinner("Retrieving results..."):
                    if search_option == "Similarity Search":
                        retriever = vectorstore.as_retriever(search_kwargs={"k": int(search_arguments)})
                        response = retriever.invoke(user_query)
                        
                    elif search_option == "Maximal Marginal Relevance search":
                        retriever = vectorstore.as_retriever(
                            search_type="mmr", 
                            search_kwargs={"k": int(search_arguments), "lambda_mult": 0.5}
                        )
                        response = retriever.invoke(user_query)
                        
                    else:  # Contextual Compression Retriever
                        base_retriever = vectorstore.as_retriever(search_kwargs={"k": int(search_arguments)})
                        compressor = LLMChainExtractor.from_llm(gpt_llm)
                        compression_retriever = ContextualCompressionRetriever(
                            base_compressor=compressor, 
                            base_retriever=base_retriever
                        )
                        response = compression_retriever.invoke(user_query)

                    # Display results
                    if response:
                        st.subheader(f"Top results ({len(response)} found):")
                        for i, doc in enumerate(response, start=1):
                            with st.expander(f"Result {i}", expanded=True):
                                st.markdown(doc.page_content)
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.caption(f"Metadata: {doc.metadata}")
                    else:
                        st.info("No results found for your query.")
            else:
                st.warning("Please enter a query to search.")