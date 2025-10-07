from langchain.schema.runnable import RunnableLambda

def word_count(x):
    return len(x.split())

x= "Hi, I am Gaurav"
    
print(word_count(x))
print((lambda x: len(x.split()))(x))
wordcount=RunnableLambda(word_count)
print(wordcount.invoke(x))