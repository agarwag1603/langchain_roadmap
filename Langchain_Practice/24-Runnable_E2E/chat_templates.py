from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template_rag = ChatPromptTemplate.from_messages (
    [
        ("system","You are an AI assistant that helps answering the questions based on PDF and web loaded data in vector store."),
        ("human","{question}"),
         MessagesPlaceholder("agent_scratchpad"), 
    ]
)

chat_template_summary = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant who summarizes the topic."),
        ("human","Please summarize the topic: \n\n {topic}")
    ]
)
