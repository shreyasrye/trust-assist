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

# Split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Generate embeddings
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in tqdm.tqdm(text_chunks, desc="Generating embeddings"):
        response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)
    return embeddings


# pdf_path = "CX50_User_Manual.pdf"
# text = extract_text_from_pdf(pdf_path)
# text_chunks = split_text(text)
# embeddings = generate_embeddings(text_chunks)

# # Store embeddings in Pinecone
# ids = [f"chunk-{i}" for i in range(len(text_chunks))]
# metadata = [{'text': chunk} for chunk in text_chunks]
# pinecone_vectors = list(zip(ids, embeddings, metadata))
# index.upsert(vectors=pinecone_vectors)


# Function to query Pinecone and generate response
def query_pinecone_and_generate_response(query):
    source_list = []
    texts = []
    
    # Generate embedding for the query
    query_embedding = client.embeddings.create(
        input=[query], 
        model="text-embedding-ada-002",
        ).data[0].embedding
    
    # Query Pinecone for the most relevant chunks
    query_result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    # Retrieve the text of the most relevant chunks
    relevant_texts = [match['metadata']['text'] for match in query_result['matches']]
    
    # Combine the relevant texts and query for context
    context = '\n\n'.join(relevant_texts)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    # Generate response with GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# Example usage
query = "How can I reboot the system?"
response = query_pinecone_and_generate_response(query)
print(response)