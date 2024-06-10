import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import json
import PyPDF2
import tqdm

# Initial API setup
with open("config.json", "r") as config_file:
    config = json.load(config_file)

client = OpenAI(api_key=config["openai"]["api_key"])
PINECONE_KEY = config["pinecone"]["api_key"]
PINECONE_ENV = config["pinecone"]["environment"]

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("trustassist")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Function to split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to generate embeddings
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in tqdm.tqdm(text_chunks, desc="Generating embeddings"):
        response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)
    return embeddings


pdf_path = "CX50_User_Manual.pdf"
text = extract_text_from_pdf(pdf_path)
text_chunks = split_text(text)
embeddings = generate_embeddings(text_chunks)

# Store embeddings in Pinecone
ids = [f"chunk-{i}" for i in range(len(text_chunks))]
metadata = [{'text': chunk} for chunk in text_chunks]
pinecone_vectors = list(zip(ids, embeddings, metadata))
index.upsert(vectors=pinecone_vectors)




