from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

def add_to_vectorstore(list_of_docs):

    # Embed and index
    embeddings = GPT4AllEmbeddings()

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=list_of_docs,
        collection_name="rag-chroma",
        embedding=embeddings,
    )

    return vectorstore.as_retriever()