# memory_system/memory_connector.py
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

from .memory_models import Memory
from .embedding_utils import compute_adaptive_similarity #, cosine_similarity_np

# Forward declaration for type hinting MemoryBlossom
class MemoryBlossom:
    pass

class MemoryConnector:
    def __init__(self, memory_blossom_instance: MemoryBlossom):
        self.memory_blossom = memory_blossom_instance
        # memory_id -> List of (connected_memory_id, similarity_score, relation_type_str)
        self.connection_graph: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
        self.memory_clusters: List[Set[str]] = [] # List of sets, each set is a cluster of memory_ids
        self.semantic_fields: Dict[str, List[str]] = {} # field_name -> list_of_memory_ids

    def _infer_relation_type(self, mem1: Memory, mem2: Memory, similarity: float) -> str:
        """Infers the type of relationship between two memories."""
        # Time-based
        time_diff_hours = abs((mem1.creation_time - mem2.creation_time).total_seconds()) / 3600
        if time_diff_hours < 1 and similarity > 0.6: # Within 1 hour and reasonably similar
            return "temporal_proximity"
        if time_diff_hours < 24 and similarity > 0.5:
            return "same_day_context"

        # Content-based (simple heuristics)
        if mem1.memory_type == mem2.memory_type and similarity > 0.75:
            return f"strong_{mem1.memory_type.lower()}_link"
        if "how to" in mem1.content.lower() and "how to" in mem2.content.lower() and similarity > 0.6:
            return "procedural_similarity"
        if mem1.emotion_score > 0.7 and mem2.emotion_score > 0.7 and similarity > 0.5:
            return "shared_high_emotion"

        # Default semantic link
        if similarity > 0.8: return "strong_semantic"
        if similarity > 0.65: return "moderate_semantic"
        if similarity > 0.5: return "weak_semantic"

        return "related" # Generic fallback

    def analyze_all_memories(self):
        """Analyzes all memories to build the connection graph and identify clusters."""
        print("MemoryConnector: Starting analysis of all memories...")
        all_memories_flat: List[Memory] = []
        for mem_list in self.memory_blossom.memory_stores.values():
            all_memories_flat.extend(mem_list)

        if not all_memories_flat:
            print("MemoryConnector: No memories to analyze.")
            return

        self.connection_graph.clear() # Reset graph

        for i in range(len(all_memories_flat)):
            mem1 = all_memories_flat[i]
            if mem1.embedding is None: continue
            for j in range(i + 1, len(all_memories_flat)):
                mem2 = all_memories_flat[j]
                if mem2.embedding is None: continue

                similarity = compute_adaptive_similarity(mem1.embedding, mem2.embedding)
                if similarity > 0.4: # Threshold for considering a connection
                    relation_type = self._infer_relation_type(mem1, mem2, similarity)
                    self.connection_graph[mem1.id].append((mem2.id, similarity, relation_type))
                    self.connection_graph[mem2.id].append((mem1.id, similarity, relation_type))
        print(f"MemoryConnector: Connection graph built. {len(self.connection_graph)} memories have connections.")
        self._detect_memory_clusters(all_memories_flat)
        self._identify_semantic_fields(all_memories_flat) # conceptual

    def analyze_specific_memory(self, new_memory: Memory):
        """Analyzes a new memory and updates connections involving it."""
        if new_memory.embedding is None: return

        print(f"MemoryConnector: Analyzing new memory {new_memory.id[:8]}...")
        # Remove old connections for this memory if it's an update (not strictly necessary for new)
        if new_memory.id in self.connection_graph:
            for connected_id, _, _ in self.connection_graph[new_memory.id]:
                if connected_id in self.connection_graph:
                    self.connection_graph[connected_id] = [
                        conn for conn in self.connection_graph[connected_id] if conn[0] != new_memory.id
                    ]
            del self.connection_graph[new_memory.id]

        all_memories_flat: List[Memory] = []
        for mem_list in self.memory_blossom.memory_stores.values():
            all_memories_flat.extend(mem_list)

        for existing_memory in all_memories_flat:
            if existing_memory.id == new_memory.id or existing_memory.embedding is None:
                continue
            similarity = compute_adaptive_similarity(new_memory.embedding, existing_memory.embedding)
            if similarity > 0.4:
                relation_type = self._infer_relation_type(new_memory, existing_memory, similarity)
                self.connection_graph[new_memory.id].append((existing_memory.id, similarity, relation_type))
                self.connection_graph[existing_memory.id].append((new_memory.id, similarity, relation_type))
        # Optionally, re-run clustering/semantic field identification if many memories are added/updated frequently
        # For a single new memory, a full re-cluster might be too much.
        # Consider incremental clustering or periodic full re-analysis.


    def _detect_memory_clusters(self, all_memories_flat: List[Memory]):
        """Detects clusters of highly interconnected memories (simplified)."""
        # This is a very simplified clustering. Real applications might use graph algorithms (e.g., Louvain).
        self.memory_clusters.clear()
        visited_ids: Set[str] = set()
        for mem_id in self.connection_graph:
            if mem_id in visited_ids:
                continue

            current_cluster: Set[str] = set()
            queue = [mem_id]
            visited_ids.add(mem_id)
            current_cluster.add(mem_id)

            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                for neighbor_id, similarity, _ in self.connection_graph.get(curr, []):
                    if neighbor_id not in visited_ids and similarity > 0.7: # Higher threshold for strong cluster
                        visited_ids.add(neighbor_id)
                        current_cluster.add(neighbor_id)
                        queue.append(neighbor_id)
            if len(current_cluster) > 1: # Only consider clusters with more than one memory
                self.memory_clusters.append(current_cluster)
        print(f"MemoryConnector: Detected {len(self.memory_clusters)} memory clusters.")

    def _identify_semantic_fields(self, all_memories_flat: List[Memory]):
        """Placeholder for identifying broader semantic fields."""
        # This would be more complex, potentially using dimensionality reduction (PCA/t-SNE)
        # then clustering in the reduced space to find "regions" of meaning.
        # For now, we can make a simplification:
        self.semantic_fields.clear()
        temp_fields = defaultdict(list)
        if not all_memories_flat or not hasattr(all_memories_flat[0], 'memory_type'): return

        for mem in all_memories_flat:
            # Group by primary memory type as a very rough proxy for semantic field
            temp_fields[mem.memory_type].append(mem.id)

        # Filter for fields with enough memories
        for field_name, mem_ids in temp_fields.items():
            if len(mem_ids) > 2: # Arbitrary threshold
                self.semantic_fields[field_name] = mem_ids
        print(f"MemoryConnector: Identified {len(self.semantic_fields)} pseudo-semantic fields (by type).")


    def enhance_retrieval(self,
                          initial_memories: List[Memory],
                          query: str, # Query used for initial retrieval
                          conversation_context: Optional[List[Dict[str,str]]] = None,
                          max_enhanced_memories: int = 20 # Total to return
                          ) -> List[Memory]:
        """Enhances a list of retrieved memories by adding connected/related ones."""
        if not initial_memories:
            return []

        enhanced_set: Dict[str, Memory] = {mem.id: mem for mem in initial_memories}
        # Keep track of scores for sorting later
        scores: Dict[str, float] = {mem.id: 1.0 for mem in initial_memories} # Initial memories have highest "base" score

        # 1. Add directly connected memories
        for mem in initial_memories:
            if mem.id in self.connection_graph:
                # Sort connections by similarity, prefer stronger connections
                sorted_connections = sorted(self.connection_graph[mem.id], key=lambda x: x[1], reverse=True)
                for connected_id, similarity, rel_type in sorted_connections:
                    if connected_id not in enhanced_set and len(enhanced_set) < max_enhanced_memories:
                        connected_mem_obj = self.memory_blossom.get_memory_by_id(connected_id)
                        if connected_mem_obj:
                            enhanced_set[connected_id] = connected_mem_obj
                            scores[connected_id] = scores.get(connected_id, 0) + similarity * 0.8 # Weight connected higher
                    if len(enhanced_set) >= max_enhanced_memories: break
            if len(enhanced_set) >= max_enhanced_memories: break

        # 2. Add memories from shared clusters
        if len(enhanced_set) < max_enhanced_memories:
            for cluster in self.memory_clusters:
                # Check if any of the initial memories are in this cluster
                if any(mem.id in cluster for mem in initial_memories):
                    for mem_id_in_cluster in cluster:
                        if mem_id_in_cluster not in enhanced_set and len(enhanced_set) < max_enhanced_memories:
                            cluster_mem_obj = self.memory_blossom.get_memory_by_id(mem_id_in_cluster)
                            if cluster_mem_obj:
                                enhanced_set[mem_id_in_cluster] = cluster_mem_obj
                                scores[mem_id_in_cluster] = scores.get(mem_id_in_cluster, 0) + 0.5 # Lower weight for cluster
                        if len(enhanced_set) >= max_enhanced_memories: break
                if len(enhanced_set) >= max_enhanced_memories: break

        # 3. Story-Based/Familiar-to-Novel Bridging (Conceptual - needs more sophisticated logic)
        # This part is more about how the *agent uses* the memories,
        # but MemoryConnector can provide candidates.
        # For now, let's try to find "bridge" memories that connect diverse initial memories.
        if len(initial_memories) > 1 and len(enhanced_set) < max_enhanced_memories:
            # Find memories that connect two *different* initial memories
            # This is computationally intensive, simplify for now.
            pass


        # Sort all collected memories by their assigned scores (descending)
        # and return the top N
        final_list = sorted(list(enhanced_set.values()), key=lambda m: scores.get(m.id, 0.0), reverse=True)

        print(f"MemoryConnector: Enhanced retrieval from {len(initial_memories)} to {len(final_list)} (max {max_enhanced_memories})")
        return final_list[:max_enhanced_memories]