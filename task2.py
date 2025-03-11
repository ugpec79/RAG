import requests
import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# Constants
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
QDRANT_DB_PATH = "qdrant_db"
COLLECTION_NAME = "wikipedia_chunks"
OLLAMA_MODEL = "llama2"  # Change to preferred model

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Qdrant Client
qdrant_client = QdrantClient(path=QDRANT_DB_PATH)

# Create collection in Qdrant
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def compute_embedding(text):
    """Generate embeddings for text using Sentence-Transformers."""
    return embedding_model.encode(text).tolist()

def search_wikipedia(query, offset=0):
    """Search Wikipedia and return paginated results."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "sroffset": offset,
        "srlimit": 5,
        "format": "json"
    }
    response = requests.get(WIKI_API_URL, params=params)
    data = response.json()
    return data["query"]["search"] if "query" in data else []

def get_wikipedia_article_by_pageid(pageid):
    """Retrieve Wikipedia article text using Page ID."""
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "extracts",
        "explaintext": True,
        "format": "json"
    }
    response = requests.get(WIKI_API_URL, params=params)
    data = response.json()
    page = data["query"]["pages"].get(str(pageid))
    return page["extract"] if page and "extract" in page else None

def chunk_text(article_text, chunk_size=500):
    """Chunk article into smaller parts for embeddings."""
    return [article_text[i:i+chunk_size] for i in range(0, len(article_text), chunk_size)]

def store_chunks_in_qdrant(chunks):
    """Store chunked text in Qdrant with computed embeddings."""
    points = [
        PointStruct(id=i, vector=compute_embedding(chunk), payload={"text": chunk})
        for i, chunk in enumerate(chunks)
    ]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} chunks in Qdrant.")

def query_qdrant(user_query, top_k=3):
    """Retrieve most relevant chunks from Qdrant."""
    query_embedding = compute_embedding(user_query)
    
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    return [result.payload["text"] for result in search_results]

def generate_response_with_ollama(user_query, retrieved_chunks):
    """Use Ollama LLM to generate a response based on retrieved text."""
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are an AI assistant using Wikipedia knowledge.
    Based on the following information, answer the question:
    
    {context}
    
    User Query: {user_query}
    """
    
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"]

def main():
    """Main function to search, retrieve, store in Qdrant, and query articles."""
    query = input("Enter search term: ").strip()
    offset = 0
    selected_pageid = None

    while not selected_pageid:
        results = search_wikipedia(query, offset)
        if not results:
            print("No results found. Try another search.")
            query = input("Enter search term: ").strip()
            offset = 0
            continue

        print("\nSearch Results:")
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result['title']} (Page ID: {result['pageid']})")

        print("n. Next results")
        print("x. Exit")

        choice = input("\nSelect an article (1-5) or enter 'n' for more results: ").strip().lower()

        if choice.isdigit() and 1 <= int(choice) <= len(results):
            selected_pageid = results[int(choice) - 1]["pageid"]
        elif choice == "n":
            offset += 5
        elif choice == "x":
            print("Exiting.")
            return
        else:
            print("Invalid choice. Try again.")

    print("\nFetching article...\n")
    article_text = get_wikipedia_article_by_pageid(selected_pageid)

    if not article_text:
        print("Failed to retrieve article.")
        return
    
    chunks = chunk_text(article_text, chunk_size=500)
    print(f"\nArticle chunked into {len(chunks)} segments.")

    store_chunks_in_qdrant(chunks)

    user_query = input("\nEnter a query to search in stored chunks: ")
    retrieved_chunks = query_qdrant(user_query)

    if retrieved_chunks:
        response = generate_response_with_ollama(user_query, retrieved_chunks)
        print("\nOllama Response:\n")
        print(response)
    else:
        print("No relevant results found in Qdrant.")

if __name__ == "__main__":
    main()
