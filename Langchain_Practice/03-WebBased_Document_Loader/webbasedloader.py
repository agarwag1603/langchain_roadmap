from langchain_community.document_loaders import WebBaseLoader
import bs4

web_loader=WebBaseLoader(web_path="https://en.wikipedia.org/wiki/Virat_Kohli",bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("vector-column-start"))))

print(web_loader.load())

