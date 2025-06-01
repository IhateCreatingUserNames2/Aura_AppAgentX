import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec


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
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        """Ensure the index exists, create if not"""
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

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
