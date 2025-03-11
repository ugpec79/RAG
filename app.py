from qdrant_client import QdrantClient

# Use Qdrant embedded mode
client = QdrantClient(path="qdrant_db")  # Creates a local database in this folder

# Check if it's working
print(client.get_collections())
