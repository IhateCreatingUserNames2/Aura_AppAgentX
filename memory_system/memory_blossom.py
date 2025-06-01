# memory_system/memory_blossom.py
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict

from .memory_models import Memory
from .embedding_utils import generate_embedding, compute_adaptive_similarity
# MemoryConnector will be imported conditionally to avoid circular dependency during initialization
# from .memory_connector import MemoryConnector


class MemoryBlossom:
    DEFAULT_MEMORY_STORE_PATH = "memory_blossom_data.json"

    def __init__(self, persistence_path: Optional[str] = None):
        self.memory_stores: Dict[str, List[Memory]] = defaultdict(list)
        # For now, MemoryConnector is initialized after MemoryBlossom
        self.memory_connector = None # To be set by set_memory_connector
        self.persistence_path = persistence_path or self.DEFAULT_MEMORY_STORE_PATH

        # Criticality parameters (can be tuned)
        self.memory_temperature: float = 0.7  # Randomness in selection
        self.coherence_bias: float = 0.6      # Favor structured memories
        self.novelty_bias: float = 0.4        # Favor unique memories

        # Meta-memory (can be expanded)
        self.memory_statistics: Dict[str, int] = defaultdict(int)
        self.memory_transitions: Dict[tuple[str, str], int] = defaultdict(int) # (from_type, to_type) -> count

        self.load_memories() # Load at startup

    def set_memory_connector(self, connector): # Type hint later when MemoryConnector is defined
        """Sets the memory connector after initialization to break circular dependency."""
        self.memory_connector = connector
        if self.memory_connector:
            print("MemoryBlossom: MemoryConnector set. Analyzing all memories.")
            self.memory_connector.analyze_all_memories()


    def add_memory(self,
                   content: str,
                   memory_type: str, # User now specifies type
                   metadata: Optional[Dict[str, Any]] = None,
                   emotion_score: float = 0.0,
                   coherence_score: float = 0.5,
                   novelty_score: float = 0.5,
                   initial_salience: float = 0.5) -> Memory:
        """
        Add a new memory to the appropriate memory store.
        The caller is responsible for determining the memory_type.
        """
        embedding = generate_embedding(content, memory_type)
        if embedding is None:
            print(f"Warning: Could not generate embedding for memory content: {content[:50]}...")
            # Optionally, decide if memory should still be added without embedding or raise error

        memory = Memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            emotion_score=emotion_score,
            embedding=embedding,
            coherence_score=coherence_score,
            novelty_score=novelty_score,
            initial_salience=initial_salience
        )
        self.memory_stores[memory_type].append(memory)
        self.memory_statistics[memory_type] += 1
        print(f"MemoryBlossom: Added '{memory.memory_type}' memory: '{memory.content[:30]}...'")

        if self.memory_connector: # If connector exists, re-analyze
             self.memory_connector.analyze_specific_memory(memory) # Or full re-analysis if cheaper

        return memory

    def retrieve_memories(self,
                          query: str,
                          target_memory_types: Optional[List[str]] = None,
                          top_k: int = 5,
                          min_similarity_threshold: float = 0.3,
                          apply_criticality: bool = True,
                          conversation_context: Optional[List[Dict[str,str]]] = None
                         ) -> List[Memory]:
        """
        Retrieve relevant memories based on query, type, and criticality.
        Now also considers conversation_context for contextual_embedding (conceptual).
        """
        query_embedding_default = generate_embedding(query, "Default")
        if query_embedding_default is None:
            return []

        candidate_memories: List[tuple[Memory, float]] = []
        search_types = target_memory_types if target_memory_types else list(self.memory_stores.keys())

        context_text = ""
        if conversation_context:
            context_text = " ".join([msg.get('content', '') for msg in conversation_context[-3:]]) # Last 3 messages

        for mem_type in search_types:
            if mem_type not in self.memory_stores:
                continue

            # Generate query embedding specific to this memory type's model for best results
            query_embedding_typed = generate_embedding(query, mem_type)
            if query_embedding_typed is None: # Fallback to default if typed model fails
                query_embedding_typed = query_embedding_default

            for memory in self.memory_stores[mem_type]:
                memory.apply_decay() # Apply decay before calculating salience for retrieval

                # Concept: Generate a contextual embedding for the memory if context is available
                # This is a placeholder for a more advanced contextualization step.
                # For now, we'll just use the primary embedding.
                # if context_text and memory.content:
                #     combined_for_contextual_embedding = f"Context: {context_text}\n\nMemory: {memory.content}"
                #     memory.contextual_embedding = generate_embedding(combined_for_contextual_embedding, mem_type)
                # else:
                #     memory.contextual_embedding = memory.embedding

                # Use primary embedding for similarity calculation for now
                similarity = compute_adaptive_similarity(query_embedding_typed, memory.embedding)

                if similarity >= min_similarity_threshold:
                    effective_salience = memory.get_effective_salience()
                    score = similarity * 0.6 + effective_salience * 0.4 # Weighted score

                    if apply_criticality:
                        # Apply criticality to memory selection (edge of chaos)
                        coherence_component = memory.coherence_score * self.coherence_bias
                        novelty_component = memory.novelty_score * self.novelty_bias
                        noise = np.random.normal(0, self.memory_temperature * 0.1) # Smaller noise
                        score = score * 0.7 + coherence_component * 0.1 + novelty_component * 0.1 + noise * 0.1
                    candidate_memories.append((memory, score))

        # Sort by combined score
        candidate_memories.sort(key=lambda x: x[1], reverse=True)
        retrieved = [mem for mem, score in candidate_memories[:top_k]]

        # Enhance with connected memories if MemoryConnector is available
        if self.memory_connector and retrieved:
            # Pass current query and context if available to enhance_retrieval
            retrieved = self.memory_connector.enhance_retrieval(retrieved, query, conversation_context)


        for mem in retrieved: # Update access only for finally retrieved memories
            mem.update_access()

        # Track memory transitions (simplified for now)
        if len(retrieved) > 1:
            self.memory_transitions[(retrieved[0].memory_type, retrieved[1].memory_type)] +=1

        return retrieved

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        for mem_list in self.memory_stores.values():
            for memory in mem_list:
                if memory.id == memory_id:
                    return memory
        return None

    def update_memory_content(self, memory_id: str, new_content: str) -> bool:
        memory = self.get_memory_by_id(memory_id)
        if memory:
            memory.content = new_content
            # Re-embed
            new_embedding = generate_embedding(new_content, memory.memory_type)
            if new_embedding is not None:
                memory.embedding = new_embedding
            memory.last_accessed = datetime.now(timezone.utc) # Treat update as access
            # Potentially re-evaluate coherence/novelty
            print(f"MemoryBlossom: Updated content for memory {memory_id}")
            return True
        return False

    def add_connection(self, mem_id1: str, mem_id2: str, strength: float, relation_type: str):
        mem1 = self.get_memory_by_id(mem_id1)
        mem2 = self.get_memory_by_id(mem_id2)
        if mem1 and mem2:
            mem1.connections.append((mem_id2, strength, relation_type))
            mem2.connections.append((mem_id1, strength, relation_type)) # Bidirectional
            print(f"MemoryBlossom: Added connection between {mem_id1[:8]} and {mem_id2[:8]}")

    def save_memories(self):
        """Saves all memory stores to a JSON file."""
        data_to_save = {
            "memory_stores": {
                mem_type: [mem.to_dict() for mem in memories]
                for mem_type, memories in self.memory_stores.items()
            },
            "memory_statistics": dict(self.memory_statistics),
            "memory_transitions": {str(k): v for k,v in self.memory_transitions.items()}, # Convert tuple keys to str
            "criticality_params": {
                "temperature": self.memory_temperature,
                "coherence_bias": self.coherence_bias,
                "novelty_bias": self.novelty_bias
            }
        }
        try:
            with open(self.persistence_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            print(f"MemoryBlossom: Saved memories to {self.persistence_path}")
        except IOError as e:
            print(f"Error saving memories to {self.persistence_path}: {e}")

    def load_memories(self):
        """Loads memories from a JSON file."""
        if not os.path.exists(self.persistence_path):
            print(f"MemoryBlossom: No persistence file found at {self.persistence_path}. Starting fresh.")
            return

        try:
            with open(self.persistence_path, 'r') as f:
                loaded_data = json.load(f)

            raw_memory_stores = loaded_data.get("memory_stores", {})
            for mem_type, mem_data_list in raw_memory_stores.items():
                self.memory_stores[mem_type] = [Memory.from_dict(data) for data in mem_data_list]

            self.memory_statistics = defaultdict(int, loaded_data.get("memory_statistics", {}))

            raw_transitions = loaded_data.get("memory_transitions", {})
            self.memory_transitions = defaultdict(int)
            for k_str, v in raw_transitions.items(): # Convert str keys back to tuples
                try:
                    # Example key: "('Explicit', 'Emotional')"
                    key_tuple = tuple(part.strip().strip("'") for part in k_str.strip("()").split(","))
                    if len(key_tuple) == 2:
                         self.memory_transitions[key_tuple] = v
                except Exception as e_parse:
                    print(f"Warning: Could not parse transition key '{k_str}': {e_parse}")


            crit_params = loaded_data.get("criticality_params", {})
            self.memory_temperature = crit_params.get("temperature", 0.7)
            self.coherence_bias = crit_params.get("coherence_bias", 0.6)
            self.novelty_bias = crit_params.get("novelty_bias", 0.4)

            print(f"MemoryBlossom: Loaded memories from {self.persistence_path}")
            # After loading, if a connector is set, it should re-analyze
            if self.memory_connector:
                print("MemoryBlossom: Re-analyzing connections for loaded memories.")
                self.memory_connector.analyze_all_memories()

        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading memories from {self.persistence_path}: {e}. Starting fresh.")
            self.memory_stores = defaultdict(list) # Reset if loading fails