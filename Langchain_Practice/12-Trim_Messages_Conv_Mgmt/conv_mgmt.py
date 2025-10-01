from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage,trim_messages, AIMessage
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

trimmer=trim_messages(
    max_tokens=70,
    strategy = "last",
    token_counter=gpt_llm,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messageprompt = [
    SystemMessage(content="You're an information assistant"),
    HumanMessage(content="Hi, I am Gaurav"),
    AIMessage(content="Hello!"),
    HumanMessage(content="I live in Singapore"),
    AIMessage(content="That is amazing to know, is it hot there?"),
    HumanMessage(content="Yes, too hot here and i like to eat dark chocolate ice cream."),
    AIMessage(content="I am sorry to hear about it"),
    HumanMessage(content="thanks"),
    AIMessage(content="You're welcome"),
]

print(trimmer.invoke(messageprompt))
print(80*"--")

prompt= ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Answer to the best you can in {language}"),
        MessagesPlaceholder(variable_name="messageprompt")
    ]
)

chain = (RunnablePassthrough.assign(messageprompt=itemgetter("messageprompt")|trimmer) | prompt | gpt_llm
)


response = chain.invoke({
    "messageprompt": messageprompt + [HumanMessage(content="which Ice cream do I prefer eating?")],
    "language": "English"
})

print(response)

