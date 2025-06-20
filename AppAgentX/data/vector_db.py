import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import logging

logger_vector_db = logging.getLogger("AppAgentX.data.vector_db")

class NodeType(Enum):
    PAGE = "page"
    ELEMENT = "element"


@dataclass
class VectorData:
    id: str
    values: List[float]
    metadata: Dict[str, Any]
    node_type: NodeType


class VectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str = "area-embedding",
        dimension: int = 2048,
        batch_size: int = 100,
    ):
        """Initialize vector store

        Args:
            api_key: Pinecone API key
            index_name: Index name
            dimension: Vector dimension
            batch_size: Batch size
        """

        logger_vector_db.info(f"VectorStore __init__: START for index '{index_name}'")
        self.pc = Pinecone(api_key=api_key)
        logger_vector_db.info(f"VectorStore __init__: Pinecone client object created for '{index_name}'.")
        self.index_name = index_name
        self.dimension = dimension  # Make sure this matches your index!
        self._ensure_index()  # This is the potentially blocking call
        self.index = self.pc.Index(self.index_name)
        logger_vector_db.info(f"VectorStore __init__: FINISHED for index '{index_name}'.")

    def _ensure_index(self):
        """Ensure the index exists, create if not"""
        logger_vector_db.info(f"VectorStore _ensure_index: START for '{self.index_name}'. Checking existence...")
        if not self.pc.has_index(self.index_name):
            logger_vector_db.info(f"VectorStore _ensure_index: Index '{self.index_name}' not found. Creating...")
            self.pc.create_index(
                name=self.index_name, dimension=self.dimension, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Match your spec
            )
            logger_vector_db.info(f"VectorStore _ensure_index: Index '{self.index_name}' creation initiated.")

        logger_vector_db.info(f"VectorStore _ensure_index: Waiting for index '{self.index_name}' to be ready...")
        wait_counter = 0
        while True:
            try:
                status = self.pc.describe_index(self.index_name).status
                if status["ready"]:
                    logger_vector_db.info(f"VectorStore _ensure_index: Index '{self.index_name}' IS READY.")
                    break
                else:
                    logger_vector_db.info(
                        f"VectorStore _ensure_index: Index '{self.index_name}' not ready, status: {status}. Waiting... ({wait_counter * 5}s)")
            except Exception as e:
                logger_vector_db.error(
                    f"VectorStore _ensure_index: Error describing index '{self.index_name}': {e}. Retrying...")

            time.sleep(5)
            wait_counter += 1
            if wait_counter > 24:  # Timeout after 2 minutes
                logger_vector_db.critical(
                    f"VectorStore _ensure_index: TIMEOUT waiting for index '{self.index_name}' after {wait_counter * 5}s.")
                raise Exception(f"Timeout waiting for Pinecone index '{self.index_name}' to become ready.")
        logger_vector_db.info(f"VectorStore _ensure_index: FINISHED for '{self.index_name}'.")

    def upsert_batch(self, vectors: List[VectorData]) -> bool:
        """Batch insert or update vectors

        Args:
            vectors: List of VectorData objects

        Returns:
            bool: Whether the operation was successful
        """
        try:
            vectors_by_type = {}
            for vec in vectors:
                if vec.node_type.value not in vectors_by_type:
                    vectors_by_type[vec.node_type.value] = []
                processed_metadata = {}
                for key, value in vec.metadata.items():
                    if isinstance(value, (dict, list)):
                        processed_metadata[key] = json.dumps(value)
                    else:
                        processed_metadata[key] = value

                vectors_by_type[vec.node_type.value].append(
                    {"id": vec.id, "values": vec.values, "metadata": processed_metadata}
                )

            for namespace, vecs in vectors_by_type.items():
                total_vectors = len(vecs)
                if total_vectors == 0:
                    continue

                if total_vectors <= self.batch_size:
                    self.index.upsert(vectors=vecs, namespace=namespace)
                else:
                    for i in range(0, total_vectors, self.batch_size):
                        batch = vecs[i : min(i + self.batch_size, total_vectors)]
                        self.index.upsert(vectors=batch, namespace=namespace)
            return True
        except Exception as e:
            print(f"Batch upsert failed: {str(e)}")
            return False

    def query_similar(
        self,
        query_vector: List[float],
        node_type: NodeType,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
    ) -> Dict:
        """Query similar vectors and parse JSON strings in the results"""
        try:
            results = self.index.query(
                namespace=node_type.value,
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                filter=filter_dict,
            )

            if "matches" in results:
                for match in results["matches"]:
                    if "metadata" in match:
                        for key, value in match["metadata"].items():
                            try:
                                if isinstance(value, str) and (
                                    value.startswith("{") or value.startswith("[")
                                ):
                                    match["metadata"][key] = json.loads(value)
                            except json.JSONDecodeError:
                                continue

            return results
        except Exception as e:
            print(f"Query failed: {str(e)}")
            return {}

    def delete_vectors(self, ids: List[str], node_type: NodeType) -> bool:
        """Delete vectors

        Args:
            ids: List of vector IDs to delete
            node_type: Node type (used as namespace)

        Returns:
            bool: Whether the operation was successful
        """
        try:
            self.index.delete(ids=ids, namespace=node_type.value)
            return True
        except Exception as e:
            print(f"Delete failed: {str(e)}")
            return False

    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"Failed to get stats: {str(e)}")
            return {}
