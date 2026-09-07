"""
VectorForge Interactive Text Search Demo

Embeds a sample text corpus into 128-dimensional vectors using pure Python/NumPy,
indexes them in VectorForge, and allows querying text statements to find the top matching pairs.

Usage:
    python scripts/demo_text_search.py "express js and node web backend framework"
"""

import sys
import hashlib
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.distance import normalize
from app.indexes.manager import IndexManager


# Expanded sample corpus of short texts across Web Dev, AI, Emotions, and Databases
SAMPLE_TEXTS = [
    # Web Development & Frameworks
    "Express.js is a minimalist web framework for Node.js used to build backend APIs.",
    "Node.js enables asynchronous JavaScript execution on the server side.",
    "React.js is a front-end library for building interactive user interfaces with components.",
    "JavaScript and TypeScript power modern full-stack web application development.",
    "FastAPI is a modern high-performance web framework for building RESTful APIs in Python.",
    "REST APIs allow clients to exchange JSON data over HTTP requests.",

    # Emotions & Psychology
    "Human emotions like empathy, joy, and sadness shape interpersonal communication.",
    "Emotional intelligence helps individuals manage stress and understand human feelings.",
    "Psychology studies human cognition, behavioral responses, and emotional well-being.",

    # AI & Machine Learning
    "Artificial intelligence and deep neural networks are transforming modern technology.",
    "Machine learning algorithms optimize predictive models using gradient descent.",
    "Natural language processing allows computers to analyze human text and speech.",
    "Computer vision models detect objects and recognize faces in digital images.",
    "Reinforcement learning agents learn optimal policies through environment interaction.",
    "PyTorch and TensorFlow are popular open-source frameworks for deep learning research.",

    # Vector DB & Search
    "Vector databases enable fast similarity search across high-dimensional embeddings.",
    "Hierarchical Navigable Small World graphs provide efficient graph-based nearest neighbor search.",
    "Inverted File indexes use K-Means clustering to partition vector spaces into Voronoi cells.",
    "Cosine similarity measures the angle between normalized vectors in multi-dimensional space.",

    # Databases & Systems
    "Relational databases use SQL queries and B-tree indexes for structured data.",
    "NoSQL document databases store unstructured data as JSON documents with horizontal scaling.",
    "Docker containers package applications with runtime dependencies for consistent deployment.",
    "Kubernetes manages container orchestration, automated scaling, and cluster health monitoring.",
    "Git version control tracks source code history and facilitates collaborative team workflows.",
]


def text_to_vector(text: str, dim: int = 128) -> np.ndarray:
    """
    Embed text into a deterministic normalized D-dimensional vector using feature hashing.
    Zero external dependencies — pure Python + NumPy!
    """
    # Replace punctuation with space and normalize
    clean_text = text.lower().replace(".", " ").replace(",", " ").replace("-", " ")
    words = clean_text.split()
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
    print("=" * 75)
    print("        VectorForge — Interactive Text Similarity Search Demo")
    print("=" * 75)

    # Initialize IndexManager
    manager = IndexManager(dimension=128)

    print(f"\nIndexing {len(SAMPLE_TEXTS)} sample texts into VectorStore...")
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
        query_text = "express js node framework"

    print(f"Query Statement: \"{query_text}\"")
    print("-" * 75)

    # Embed query statement
    query_vec = text_to_vector(query_text, dim=128)

    # Search across indexes
    for index_name in ["brute", "ivf", "hnsw"]:
        results = manager.search(query_vec, k=3, index=index_name)
        print(f"\n--- [{index_name.upper()} INDEX RESULTS] ---")
        for rank, res in enumerate(results, 1):
            doc = manager.store.get(res.id)
            text_str = doc["metadata"]["text"]
            match_status = "Exact / High Match" if res.score >= 0.45 else "Closest Relative Match"
            print(f"  {rank}. [{res.score:.4f}] ({match_status})")
            print(f"     \"{text_str}\"")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
