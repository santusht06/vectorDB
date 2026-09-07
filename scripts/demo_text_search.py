"""
VectorForge Interactive Text Search Demo

Embeds a sample text corpus into 128-dimensional vectors using pure Python/NumPy,
indexes them in VectorForge, and allows querying text statements to find the top matching pairs.

Usage:
    python scripts/demo_text_search.py "machine learning and artificial intelligence"
"""

import sys
import hashlib
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.distance import normalize
from app.indexes.manager import IndexManager


# Sample corpus of short texts across various topics
SAMPLE_TEXTS = [
    "Artificial intelligence and deep neural networks are transforming technology.",
    "Machine learning algorithms optimize models using gradient descent optimization.",
    "Natural language processing allows computers to analyze human text and speech.",
    "Computer vision models detect objects and recognize faces in digital images.",
    "Reinforcement learning agents learn optimal policies through environment interaction.",
    "Vector databases enable fast similarity search across high-dimensional embeddings.",
    "Hierarchical Navigable Small World graphs provide efficient graph-based nearest neighbor search.",
    "Inverted File indexes use K-Means clustering to partition vector spaces into Voronoi cells.",
    "Cosine similarity measures the angle between normalized vectors in multi-dimensional space.",
    "Relational databases use SQL queries and B-tree indexes for structured transactional data.",
    "NoSQL document databases store unstructured data as JSON documents with horizontal scaling.",
    "Cloud computing platforms offer scalable virtual machines, storage, and serverless functions.",
    "Distributed computing systems handle large-scale data processing across multi-node clusters.",
    "Microservice architecture decouples monolithic applications into independently deployable services.",
    "Docker containers package applications with runtime dependencies for consistent deployment.",
    "Kubernetes manages container orchestration, automated scaling, and cluster health monitoring.",
    "Python is a popular programming language widely used in data science and AI development.",
    "FastAPI is a modern high-performance web framework for building RESTful APIs in Python.",
    "NumPy provides high-performance N-dimensional array objects and mathematical operations.",
    "PyTorch and TensorFlow are popular open-source frameworks for deep learning research.",
    "Software engineering principles emphasize clean code, modular design, and comprehensive testing.",
    "Continuous integration and continuous deployment pipelines automate software testing and release.",
    "Git version control tracks source code history and facilitates collaborative team workflows.",
    "Cybersecurity protocols protect sensitive data against unauthorized network intrusion and malware.",
    "Data structure efficiency determines algorithm runtime complexity and memory footprint.",
]


def text_to_vector(text: str, dim: int = 128) -> np.ndarray:
    """
    Embed text into a deterministic normalized D-dimensional vector using feature hashing.
    Zero external dependencies — pure Python + NumPy!
    """
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)

    for word in words:
        # Hash word to index and sign
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += sign

    # Add bigram features for context
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += 1.5 * sign

    return normalize(vec)


def main():
    print("=" * 70)
    print("        VectorForge — Interactive Text Similarity Search Demo")
    print("=" * 70)

    # Initialize IndexManager
    manager = IndexManager(dimension=128)

    print(f"\nEmbedding and indexing {len(SAMPLE_TEXTS)} sample texts...")
    for idx, text in enumerate(SAMPLE_TEXTS):
        vec = text_to_vector(text, dim=128)
        manager.insert(
            vector_id=f"text_{idx}",
            vector=vec,
            metadata={"text": text},
        )

    # Build approximate indexes
    print("Building Brute Force, IVF-Flat, and HNSW indexes...")
    manager.build_all()
    print("Indexes built successfully!\n")

    # Get search statement from command line argument or prompt
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = "neural networks and machine learning models"

    print(f"Query Statement: \"{query_text}\"")
    print("-" * 70)

    # Embed query statement
    query_vec = text_to_vector(query_text, dim=128)

    # Search across indexes
    for index_name in ["brute", "ivf", "hnsw"]:
        results = manager.search(query_vec, k=3, index=index_name)
        print(f"\n--- [{index_name.upper()} INDEX RESULTS] ---")
        for rank, res in enumerate(results, 1):
            doc = manager.store.get(res.id)
            text_str = doc["metadata"]["text"]
            print(f"  {rank}. [Score: {res.score:.4f}] {text_str}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
