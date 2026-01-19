
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
os.getenv('hf_token')
from langgraph.prebuilt import create_react_agent

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llmg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))

docs = [
    "LangChain helps orchestrate prompts and models.",
    "Text splitters break large documents into chunks for retrieval.",
    "Llama models can be used with LangChain for QA."
]

documents = [Document(page_content=d) for d in docs]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

context = "\n\n".join(c.page_content for c in chunks)

prompt = ChatPromptTemplate.from_template(
    """
You are an assistant that answers questions strictly using the provided context.
Context:
{context}

Question:
{question}

If the answer is not in the context, say: "get out of my motherland".
"""
)

chain=prompt|llmg 
res=chain.invoke({
    "context":context,
    "question":"what is significance of shiva  in Langchain?"
})
print(res.content)

from langchain_text_splitters import RecursiveCharacterTextSplitter 
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=['\n\n','\n',"",""]
)
chunk=text_splitter.split_documents(documents)
context="\n\n".join(chu.page_content for chu in chunk)
print("-----------------------------------------------------------------------")
print(context)
res=chain.invoke({
    "context":context,
    "question":"how is text splitters related to langchain?"
})
print(res.content)