from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings  
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="NL-to-SQL Matching API")

# Load the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Set it in the environment variables.")

# Load the CSV file
csv_path = '/Users/abduljawadkhan/Downloads/ml-projects/ml-projects/t2sql/classification_nl_output.csv'  # Update to your file path
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

# Fill NaN values
df.fillna('', inplace=True)

# Extract texts
texts_col1 = df['Natural Language Query'].tolist()
texts_col2 = df['Alternatives'].tolist()
texts_col3 = df['Steps'].tolist()

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model='text-embedding-3-small')

def generate_embeddings(texts):
    """Generate normalized embeddings using LangChain."""
    embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(embeddings).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

# Generate embeddings for columns
embeddings_col1 = generate_embeddings(texts_col1)
embeddings_col2 = generate_embeddings(texts_col2)
embeddings_col3 = generate_embeddings(texts_col3)

# Create FAISS indices
embedding_dim = embeddings_col1.shape[1]
index_col1 = faiss.IndexFlatIP(embedding_dim)
index_col2 = faiss.IndexFlatIP(embedding_dim)
index_col3 = faiss.IndexFlatIP(embedding_dim)

index_col1.add(embeddings_col1)
index_col2.add(embeddings_col2)
index_col3.add(embeddings_col3)

class UserQuery(BaseModel):
    """Schema for user query input."""
    query: str
    top_k: int = Query(1, ge=1, le=10, description="Number of top matches to retrieve")

@app.post("/query", response_model=dict)
def find_best_match(user_query: UserQuery):
    """Find the best matching SQL query."""
    query_embedding = embeddings_model.embed_query(user_query.query)
    query_embedding = np.array(query_embedding).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    k = min(user_query.top_k, len(df))

    # Perform similarity search
    D_col1, I_col1 = index_col1.search(np.array([query_embedding]), k)
    D_col2, I_col2 = index_col2.search(np.array([query_embedding]), k)
    D_col3, I_col3 = index_col3.search(np.array([query_embedding]), k)

    # Aggregate similarities
    similarities_col1 = np.zeros(len(df))
    similarities_col2 = np.zeros(len(df))
    similarities_col3 = np.zeros(len(df))

    for idx, sim in zip(I_col1[0], D_col1[0]):
        similarities_col1[idx] = sim
    for idx, sim in zip(I_col2[0], D_col2[0]):
        similarities_col2[idx] = sim
    for idx, sim in zip(I_col3[0], D_col3[0]):
        similarities_col3[idx] = sim

    average_similarities = (similarities_col1 + similarities_col2 + similarities_col3) / 3

    # Retrieve top matches
    top_indices = np.argsort(-average_similarities)[:k]
    matches = df.iloc[top_indices].to_dict(orient='records')

    return {
        "query": user_query.query,
        "matches": matches
    }