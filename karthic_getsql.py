import os
import pandas as pd
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings  
import faiss
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    openai_api_key = input("Enter your OpenAI API key: ")

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('/Users/abduljawadkhan/Downloads/ml-projects/ml-projects/t2sql/classification_nl_output.csv')  # Replace with your CSV file path

# Fill NaN values with empty strings
df['Natural Language Query'] = df['Natural Language Query'].fillna('')
df['Alternatives'] = df['Alternatives'].fillna('')
df['Steps'] = df['Steps'].fillna('')

# Extract texts from the last three columns
texts_col1 = df['Natural Language Query'].tolist()
texts_col2 = df['Alternatives'].tolist()
texts_col3 = df['Steps'].tolist()

# Initialize the OpenAIEmbeddings object from LangChain
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model='text-embedding-3-small')

# Function to generate embeddings using LangChain
def generate_embeddings(texts):
    embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(embeddings).astype('float32')
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

# Generate and normalize embeddings for each column
print("Generating embeddings for 'Natural Language Query'...")
embeddings_col1 = generate_embeddings(texts_col1)

print("Generating embeddings for 'Alternatives'...")
embeddings_col2 = generate_embeddings(texts_col2)

print("Generating embeddings for 'Steps'...")
embeddings_col3 = generate_embeddings(texts_col3)

# Get the embedding dimension
embedding_dim = embeddings_col1.shape[1]

# Create FAISS indices for each set of embeddings
index_col1 = faiss.IndexFlatIP(embedding_dim)
index_col2 = faiss.IndexFlatIP(embedding_dim)
index_col3 = faiss.IndexFlatIP(embedding_dim)

# Add embeddings to the indices
index_col1.add(embeddings_col1)
index_col2.add(embeddings_col2)
index_col3.add(embeddings_col3)

def get_top_matches(user_query, k=3):
    """
    Get the top k matches for a user query and return the results as a list.
    
    Args:
        user_query (str): The natural language query to match against.
        k (int): The number of top matches to retrieve. Default is 3.
    
    Returns:
        list: A list of dictionaries containing information about the top matches.
    """
    # Get and normalize the query embedding using LangChain
    query_embedding = embeddings_model.embed_query(user_query)
    query_embedding = np.array(query_embedding).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Perform similarity search in each index
    D_col1, I_col1 = index_col1.search(np.array([query_embedding]), k)
    D_col2, I_col2 = index_col2.search(np.array([query_embedding]), k)
    D_col3, I_col3 = index_col3.search(np.array([query_embedding]), k)

    # Initialize arrays to hold similarities
    num_rows = len(df)
    similarities_col1 = np.zeros(num_rows)
    similarities_col2 = np.zeros(num_rows)
    similarities_col3 = np.zeros(num_rows)

    # Map similarities back to the DataFrame rows
    for idx, sim in zip(I_col1[0], D_col1[0]):
        similarities_col1[idx] = sim

    for idx, sim in zip(I_col2[0], D_col2[0]):
        similarities_col2[idx] = sim

    for idx, sim in zip(I_col3[0], D_col3[0]):
        similarities_col3[idx] = sim

    # Compute the average similarity per row
    average_similarities = (similarities_col1 + similarities_col2 + similarities_col3) / 3

    # Get the indices of the top k matches
    top_k_indices = np.argsort(average_similarities)[-k:][::-1]

    # Retrieve information for the top k matches
    top_matches = []
    for idx in top_k_indices:
        row = df.iloc[idx]
        top_matches.append({
            'SQL Query': row['SQL Query'],
            'Type': row['Type'],
            'Stakeholders': row['Stakeholders'],
            'Natural Language Query': row['Natural Language Query'],
            'Alternatives': row['Alternatives'],
            'Steps': row['Steps'],
            'Filtered Schema': row.get('Filtered Schema', '')  # Safely handle missing column
        })

    return top_matches

# Example usage:
if __name__ == "__main__":
    # Prompt the user for input
    user_query = input("\nEnter your natural language question (or type 'exit' or 'quit' to stop): ")
    
    if user_query.lower() not in ['exit', 'quit']:
        top_matches = get_top_matches(user_query, k=3)

        # Print the top matches
        print("\nTop Matches:")
        for i, match in enumerate(top_matches, start=1):
            print(f"\nMatch {i}:")
            for key, value in match.items():
                print(f"{key}: {value}")