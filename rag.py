from pinecone import Pinecone
from openai import OpenAI
import json
import PyPDF2
import tqdm
import prototype

with open("config.json", "r") as config_file:
        config = json.load(config_file)

client = OpenAI(api_key=config["openai"]["api_key"])

PINECONE_KEY = config["pinecone"]["api_key"]
PINECONE_ENV = config["pinecone"]["environment"]
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("trustassist")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def split_text(text, chunk_size=500):
    """
    Splits a long text into smaller chunks.
    """
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def generate_embeddings(text_chunks):
    """
    Generates embeddings for text chunks using OpenAI's text-embedding-ada-002 model.
    """
    embeddings = []
    for chunk in tqdm.tqdm(text_chunks, desc="Generating embeddings"):
        response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)
    return embeddings

def query_pinecone_and_generate_response(query):
    """
    Queries the Pinecone index with a given query and generates a response using OpenAI's gpt-4o model.
    """
    query_embedding = client.embeddings.create(
        input=[query], 
        model="text-embedding-ada-002",
    ).data[0].embedding
    
    query_result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    relevant_texts = [match['metadata']['text'] for match in query_result['matches']]
    
    context = '\n\n'.join(relevant_texts)
    prompt = f"Context: {context}\n\n Create a set of troubleshooting steps for the hospital staff to solve the issue. Question: {query}.\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def main(input_query):
    pdf_path = config["pdf_path"]

    text = extract_text_from_pdf(pdf_path)

    # Split text into smaller chunks
    text_chunks = split_text(text)

    embeddings = generate_embeddings(text_chunks)

    ids = [f"chunk-{i}" for i in range(len(text_chunks))]
    metadata = [{'text': chunk} for chunk in text_chunks]
    pinecone_vectors = list(zip(ids, embeddings, metadata))

    index.upsert(vectors=pinecone_vectors)

    # Query Pinecone index and generate response
    
    response = query_pinecone_and_generate_response(input_query)
    return response

# if __name__ == "__main__":
#     context = json.loads(prototype.main())
#     print(context)

#     query = context["posed_question"]
#     print(main(query))