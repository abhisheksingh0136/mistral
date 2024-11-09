import time
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# Store FAISS index and timestamp in memory
vectorstore_cache = {"index": None, "timestamp": None}
index_age_limit = 3600  # 1 hour in seconds

# Fixed URLs from backend
fixed_urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]

def load_or_create_faiss_index(doc_splits):
    current_time = time.time()
    
    # Check if the in-memory index is younger than 1 hour
    if vectorstore_cache["index"] and (current_time - vectorstore_cache["timestamp"]) < index_age_limit:
        vectorstore = vectorstore_cache["index"]
    else:
        # Create a new FAISS index and store it in memory
        embedding_model = embeddings.OllamaEmbeddings(model='nomic-embed-text')
        vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embedding_model)
        
        # Update cache
        vectorstore_cache["index"] = vectorstore
        vectorstore_cache["timestamp"] = current_time
    
    return vectorstore

def process_input(question):
    model_local = ChatOllama(model="mistral")

    # Load documents from fixed URLs
    docs = [WebBaseLoader(url).load() for url in fixed_urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    # Load or create the FAISS index in memory
    vectorstore = load_or_create_faiss_index(doc_splits)
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Streamlit UI
st.title("Document Query with Ollama")
st.write("Ask a question to query the fixed set of documents.")

# Text input for Question only
question = st.text_input("Question")

# Button to process the query
if st.button("Submit"):
    if question:
        with st.spinner("Processing..."):
            answer = process_input(question)
        st.write("Answer:", answer)
    else:
        st.warning("Please enter a question.")
