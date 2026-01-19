

# # from operator import itemgetter
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_core.documents import Document
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# # from langchain_core.runnables.history import RunnableWithMessageHistory
# # from langchain_community.chat_message_histories import ChatMessageHistory
# # from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
# # from langchain_community.vectorstores import FAISS

# # import os
# # from dotenv import load_dotenv
# # import streamlit as st 
# # load_dotenv()
# # os.getenv('hf_token')
# # from langgraph.prebuilt import create_react_agent

# # from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# # llmg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))

# # docs = [
# #     "LangChain helps orchestrate prompts and models.",
# #     "Text splitters break large documents into chunks for retrieval.",
# #     "Llama models can be used with LangChain for QA."
# # ]
# # documents = [Document(page_content=d) for d in docs]


# # splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# # chunks = splitter.split_documents(documents)
# # emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # vs = FAISS.from_documents(chunks, embedding=emb)
# # retriever = vs.as_retriever(search_kwargs={"k": 4})

# # prompt = ChatPromptTemplate.from_messages([
# #     ("system", "You answer strictly using the provided context."),
# #     MessagesPlaceholder("chat_history"),
# #     ("human",
# #      "Context:\n{context}\n\nQuestion:\n{question}\n\n"
# #      "If the answer is not in the context, say: \"get out of my motherland\".")
# # ])

# # join_docs = RunnableLambda(lambda ds: "\n\n".join(d.page_content for d in ds))

# # chain = (
# #     RunnableParallel(
# #         context=itemgetter("question") | retriever | join_docs,
# #         question=itemgetter("question"),
# #         chat_history=itemgetter("chat_history")
# #     )
# #     | prompt
# #     | llmg
# # )

# # store = {}
# # def get_session_history(session_id: str):
# #     if session_id not in store:
# #         store[session_id] = ChatMessageHistory()
# #     return store[session_id]

# # chain_with_history = RunnableWithMessageHistory(
# #     chain,
# #     get_session_history=get_session_history,
# #     input_messages_key="question",
# #     history_messages_key="chat_history",
# # )

# # cfg = {"configurable": {"session_id": "user-1"}}

# # res1 = chain_with_history.invoke({"question": "what is the significance of text splitters in langchain?"}, config=cfg)
# # print(res1.content)

# # res2 = chain_with_history.invoke({"question": "and how does that help retrieval?"}, config=cfg)
# # print(res2.content)
# # res2 = chain_with_history.invoke({"question": "what is my previous question about?"}, config=cfg)
# # print(res2.content)


# # res3 = chain_with_history.invoke({"question": "tell me about quantum teleportation details"}, config=cfg)
# # print(res3.content)

# import os
# from dotenv import load_dotenv
# load_dotenv()
# os.getenv('hf_token')
# from langgraph.prebuilt import create_react_agent

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# llmg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))
# from operator import itemgetter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnableParallel, RunnableLambda
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
# import numpy as np


# docs = [
#     "LangChain helps orchestrate prompts and models.",
#     "Text splitters break large documents into chunks for retrieval.",
#     "Llama models can be used with LangChain for QA."
# ]
# documents = [Document(page_content=d) for d in docs]

# splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# chunks = splitter.split_documents(documents)

# emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# chunk_texts = [c.page_content for c in chunks]
# doc_vecs = emb.embed_documents(chunk_texts)
# doc_mat = np.asarray(doc_vecs, dtype=np.float32)
# doc_norms = np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12
# doc_mat_norm = doc_mat / doc_norms

# def numpy_retrieve(question: str, k: int = 4):
#     q = np.asarray(emb.embed_query(question), dtype=np.float32)
#     q = q / (np.linalg.norm(q) + 1e-12)
#     sims = doc_mat_norm @ q
#     idx = np.argsort(-sims)[:k]
#     return [chunks[i] for i in idx]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You answer strictly using the provided context."),
#     MessagesPlaceholder("chat_history"),
#     ("human",
#      "Context:\n{context}\n\nQuestion:\n{question}\n\n"
#      "If the answer is not in the context, say: \"get out of my motherland\".")
# ])

# join_docs = RunnableLambda(lambda ds: "\n\n".join(d.page_content for d in ds))

# chain = (
#     RunnableParallel(
#         context=itemgetter("question") | RunnableLambda(lambda q: numpy_retrieve(q, k=4)) | join_docs,
#         question=itemgetter("question"),
#         chat_history=itemgetter("chat_history"),
#     )
#     | prompt
#     | llmg
# )

# store = {}
# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# chain_with_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history=get_session_history,
#     input_messages_key="question",
#     history_messages_key="chat_history",
# )

# cfg = {"configurable": {"session_id": "user-1"}}

# res1 = chain_with_history.invoke({"question": "what is the significance of text splitters in langchain?"}, config=cfg)
# print("Q1:", res1.content)

# res2 = chain_with_history.invoke({"question": "and how does that help retrieval?"}, config=cfg)
# print("Q2:", res2.content)

# res3 = chain_with_history.invoke({"question": "what is my previous question about?"}, config=cfg)
# print("Q3:", res3.content)

# res4 = chain_with_history.invoke({"question": "tell me about quantum teleportation details"}, config=cfg)
# print("Q4:", res4.content)



from operator import itemgetter
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
import streamlit as st
hf_token=st.secrets["HF_TOKEN"]

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llmg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))


docs = [
    "Steps to reset your password...",
    "How to fix VPN connection...",
    "Guide to unlock user account...",
    "Troubleshooting two‑factor authentication not working...",
    "How to update your email address in the account settings...",
    "Why I am not receiving the password reset email...",
    "How to fix slow internet or network latency issues...",
    "Troubleshooting application crash on startup...",
    "How to clear cache and cookies for browser issues...",
    "Steps to resolve session timeout errors while logging in...",
    "How to fix mobile app login issues on Android or iOS...",
    "Troubleshooting access denied error for restricted pages...",
    "How to resolve payment failure during checkout...",
    "What to do when the system shows incorrect password error...",
    "How to change your username or display name...",
    "Troubleshooting server unreachable or 500 error...",
    "How to fix device not registered error...",
    "Guide to enable or disable VPN split tunneling...",
    "How to restore deleted files from cloud storage...",
    "Steps to troubleshoot email syncing issues...",
    "Troubleshooting slow loading dashboard or UI...",
    "How to fix authentication token expired error...",
    "Resolving license expired or subscription inactive issues...",
    "How to reinstall the desktop client to fix corrupted files...",
    "Troubleshooting microphone not detected in meetings...",
    "How to fix webcam not working issues...",
    "Steps to resolve continuous logout or auto‑logout issues...",
    "Troubleshooting high CPU usage caused by the app...",
    "How to check system requirements and compatibility..."
]

documents = [Document(page_content=d) for d in docs]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chunk_texts = [c.page_content for c in chunks]
doc_vecs = emb.embed_documents(chunk_texts)
doc_mat = np.asarray(doc_vecs, dtype=np.float32)
doc_norms = np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12
doc_mat_norm = doc_mat / doc_norms

def numpy_retrieve(question: str, k: int = 4):
    q = np.asarray(emb.embed_query(question), dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = doc_mat_norm @ q
    idx = np.argsort(-sims)[:k]
    return [chunks[i] for i in idx]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer strictly using the provided context."),
    MessagesPlaceholder("chat_history"),
    ("human",
     "Context:\n{context}\n\nQuestion:\n{question}\n\n"
     "If the answer is not in the context, say: \"get out of my motherland\".")
])

join_docs = RunnableLambda(lambda ds: "\n\n".join(d.page_content for d in ds))
chain = (prompt | llmg)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def send(question: str, session_id: str = "user-1"):
    history = get_session_history(session_id)
    chat_history = history.messages
    context_docs = numpy_retrieve(question, k=3)
    context = join_docs.invoke(context_docs)
    res = chain.invoke({
        "question": question,
        "chat_history": chat_history,
        "context": context
    })
    history.add_user_message(question)
    history.add_ai_message(res.content)
    return res, context_docs




st.set_page_config(page_title="Smart Support Assistant")
st.title("Smart Support Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user-1"

text = st.text_area("Enter your issue:", height=150)
submit = st.button("Submit")

if submit and text.strip():
    res, ctx_docs = send(text, st.session_state.session_id)
    st.subheader("Answer")
    st.write(res.content)

st.subheader("Chat History")
history = get_session_history(st.session_state.session_id).messages
for m in history:
    if m.type == "human":
        st.markdown(f"**You:** {m.content}")
    else:
        st.markdown(f"**Assistant:** {m.content}")


