# memory_system/embedding_utils.py
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional, List

# Global cache for SentenceTransformer models
_model_cache: Dict[str, SentenceTransformer] = {}

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A good general-purpose small model

# --- Model Configuration (can be externalized to a config file) ---
# Based on your MemoryBlossom description
EMBEDDING_MODELS_CONFIG = {
    "Explicit": "BAAI/bge-small-en-v1.5",        # Factual, precise
    "Emotional": "all-MiniLM-L6-v2",             # General, captures affect well enough for a start
                                                 # instructor-xl is large, start smaller
    "Procedural": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", # Good for how-to, steps
    "Flashbulb": "nomic-ai/nomic-embed-text-v1.5", # High significance
    "Somatic": "clip-ViT-B-32",                  # For multimodal, but use text part for now
                                                 # If SentenceTransformer can't load CLIP directly,
                                                 # you might need a different library or a text-proxy.
                                                 # For simplicity, we'll use a text model if CLIP fails.
    "Liminal": "mixedbread-ai/mxbai-embed-large-v1", # High entropy, exploratory
    "Generative": "all-MiniLM-L6-v2",            # Creative content
    "Default": DEFAULT_EMBEDDING_MODEL
}

def get_embedding_model(model_name_or_type: str) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model from cache or downloads it.
    Can take a memory type (e.g., "Emotional") or a direct model name.
    """
    resolved_model_name = EMBEDDING_MODELS_CONFIG.get(model_name_or_type, model_name_or_type)
    if resolved_model_name not in _model_cache:
        print(f"Loading embedding model: {resolved_model_name} (for type/name: {model_name_or_type})")
        try:
            # nomic-embed-text requires trust_remote_code=True
            trust_code = "nomic-ai" in resolved_model_name
            _model_cache[resolved_model_name] = SentenceTransformer(resolved_model_name, trust_remote_code=trust_code)
        except Exception as e:
            print(f"Warning: Could not load model {resolved_model_name}. Error: {e}. Falling back to {DEFAULT_EMBEDDING_MODEL}.")
            if DEFAULT_EMBEDDING_MODEL not in _model_cache: # Load default if not already loaded
                 _model_cache[DEFAULT_EMBEDDING_MODEL] = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            _model_cache[resolved_model_name] = _model_cache[DEFAULT_EMBEDDING_MODEL] # Assign default to failed model name
    return _model_cache[resolved_model_name]

def generate_embedding(text: str, memory_type: Optional[str] = "Default") -> Optional[np.ndarray]:
    """Generates an embedding for the given text using the specified memory_type's model."""
    try:
        model_key = memory_type if memory_type in EMBEDDING_MODELS_CONFIG else "Default"
        model = get_embedding_model(model_key)
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding for type '{memory_type}': {e}")
        return None

def cosine_similarity_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two numpy vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    # Ensure they are 2D for sklearn's cosine_similarity
    vec1_2d = vec1.reshape(1, -1)
    vec2_2d = vec2.reshape(1, -1)
    # sklearn's cosine_similarity returns a 2D array, e.g., [[similarity]]
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def compute_adaptive_similarity(embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> float:
    """
    Compute similarity between embeddings, handling potential None values and dimension differences.
    This is a simplified version. A more advanced one might try to project to a common space
    if dimensions differ significantly and models are known.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0  # Or some other default for missing embeddings

    if embedding1.shape[0] != embedding2.shape[0]:
        # print(f"Warning: Comparing embeddings of different dimensions: {embedding1.shape[0]} vs {embedding2.shape[0]}. Truncating.")
        min_dim = min(embedding1.shape[0], embedding2.shape[0])
        truncated_emb1 = embedding1[:min_dim]
        truncated_emb2 = embedding2[:min_dim]
        # Apply a penalty for dimension mismatch
        dim_difference_ratio = abs(embedding1.shape[0] - embedding2.shape[0]) / max(embedding1.shape[0], embedding2.shape[0])
        similarity_penalty = 0.1 * dim_difference_ratio # e.g. 10% penalty factor for difference
        return max(0.0, cosine_similarity_np(truncated_emb1, truncated_emb2) - similarity_penalty)

    return cosine_similarity_np(embedding1, embedding2)