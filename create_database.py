from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil
import json

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
MD_DATA_PATH = "data/markdown"
JSON_DATA_PATH = "data/json/test-bank.json"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    # json_documents = load_json_documents(JSON_DATA_PATH)
    # documents.extend(json_documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(MD_DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# def load_json_documents(file_path):
#     documents = []
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#         for item in data:
#             # Combine question and answers into a single string
#             question = item['question']
#             answers = '\n'.join([f"{key}: {value}" for key, value in item['answers'].items()])
#             content = f"Q: {question}\nAnswers:\n{answers}\nCorrect Answer: {item['correct_answer']}"
#             documents.append(Document(page_content=content, metadata={"source": "json"}))
#     return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
