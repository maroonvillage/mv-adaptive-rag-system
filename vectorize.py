import uuid
import chromadb
from chromadb.config import Settings
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

def add_to_vectorstore_client(list_of_docs):

    # Embed and index
    embeddings = GPT4AllEmbeddings()

    client = chromadb.HttpClient(host="http://host.docker.internal:8000",
                                 settings=Settings(allow_reset=True))
    client.reset()  # resets the database
    collection = client.create_collection("rag-chroma")
    for doc in list_of_docs:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )

    # tell LangChain to use our client and collection name
    vectorstore = Chroma(
        client=client,
        collection_name="rag-chroma",
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever()