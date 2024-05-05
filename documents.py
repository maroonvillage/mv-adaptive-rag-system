import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def get_documents():
    # Define the directory path
    directory_path = "pdfs"

    # Get a list of files in the directory
    files = os.listdir(directory_path)

    pdf_paths = []
    # Print the list of files
    #print("Files in the directory:")
    for file in files:
        if file.endswith(".pdf"):
            pdf_paths.append(f"pdfs/{file}")

    return pdf_paths


def split_docs(pdfs_array):

    docs = [PyPDFLoader(pdf).load() for pdf in pdfs_array]

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)


    return doc_splits