from neo4j import GraphDatabase
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json


class Neo4jDatabase:
    def __init__(self, uri: str, auth: tuple):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.verify_connectivity()

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def update_node_property(
        self,
        node_id: str,
        property_name: str,
        property_value: Any,
        node_type: Optional[str] = None,  # "Page", "Element", "Action" or None
    ) -> bool:
        """Update a node property by node ID

        Args:
            node_id: Node ID value
            property_name: Name of the property to modify
            property_value: New value for the property
            node_type: Node type, such as "Page", "Element", "Action".
                      If None, will try to find any node with that ID

        Returns:
            bool: Whether the operation was successful
        """
        # Handle special property value types
        if isinstance(property_value, dict) or isinstance(property_value, list):
            property_value = json.dumps(property_value)

        if node_type:
            # Determine ID field name
            id_field = node_type.lower() + "_id"
            query = f"""
            MATCH (n:{node_type}) 
            WHERE n.{id_field} = $node_id
            SET n.{property_name} = $property_value
            RETURN n
            """
        else:
            # Try to find any node with that ID
            query = """
            MATCH (n) 
            WHERE n.page_id = $node_id OR n.element_id = $node_id OR n.action_id = $node_id
            SET n[$property_name] = $property_value
            RETURN n
            """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    query,
                    node_id=node_id,
                    property_value=property_value,
                    property_name=property_name,
                )
                record = result.single()
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to update property {property_name} for node with ID {node_id}"
                    )
                return success
        except Exception as e:
            print(f"Error updating node property: {str(e)}")
            return False

    def create_page(self, properties: Dict[str, Any]) -> str:
        """Create a Page node

        Args:
            properties: Dictionary containing page_id and other attributes
        """
        required_fields = ["page_id"]
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        properties["timestamp"] = properties.get(
            "timestamp", int(datetime.now().timestamp())
        )

        # Ensure visual_embedding_id is a string
        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])

        # Convert other_info to JSON string
        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        return self.create_node("Page", properties)

    def create_element(self, properties: Dict[str, Any]) -> str:
        """Create an Element node

        Args:
            properties: Dictionary containing element_id, element_type and other attributes
            Visual features related:
                visual_embedding_id: Element screenshot embedding ID in vector database
                visual_features: Text description of element visual features
                visual_similarity_ids: List of visually similar elements' IDs in vector index
                bounding_box: Element bounding box coordinates on the page
                ocr_info: Text information recognized by OCR
                icon_type: Icon type recognition result
        """
        required_fields = ["element_id"]
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        # Ensure visual_embedding_id is a string
        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])

        # Convert other_info to JSON string
        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        return self.create_node("Element", properties)

    def create_action(self, properties: Dict[str, Any]) -> str:
        """Create an Action node (high-level/composite action)

        Args:
            properties: Dictionary containing action_id, description, element_sequence, etc.
        """
        required_fields = ["action_id"]  # Only action_id is required
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        # Convert element_sequence to JSON string
        if "element_sequence" in properties and isinstance(
            properties["element_sequence"], list
        ):
            properties["element_sequence"] = json.dumps(properties["element_sequence"])

        return self.create_node("Action", properties)

    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Generic node creation function"""
        query = f"CREATE (n:{label} $properties) " "RETURN elementId(n) as node_id"

        with self.driver.session(database="neo4j") as session:
            result = session.run(query, properties=properties)
            record = result.single()
            return str(record["node_id"]) if record else None

    def add_element_to_page(self, page_id: str, element_id: str) -> bool:
        """Create Page-HAS_ELEMENT->Element relationship"""
        query = """
        MATCH (p:Page {page_id: $page_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (p)-[r:HAS_ELEMENT]->(e)
        RETURN type(r) as rel_type
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, page_id=page_id, element_id=element_id)
                record = result.single()
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to create HAS_ELEMENT relationship between page {page_id} and element {element_id}"
                    )
                return success
        except Exception as e:
            print(f"Error creating HAS_ELEMENT relationship: {str(e)}")
            return False

    def add_element_to_action(
        self,
        action_id: str,
        element_id: str,
        order: int,
        atomic_action: str,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create Action-COMPOSED_OF->Element relationship"""
        query = """
        MATCH (a:Action {action_id: $action_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (a)-[r:COMPOSED_OF {
            order: $order,
            atomic_action: $atomic_action,
            action_params: $action_params
        }]->(e)
        RETURN type(r) as rel_type
        """

        # Serialize action_params to JSON string
        if action_params:
            action_params = json.dumps(action_params)

        with self.driver.session(database="neo4j") as session:
            result = session.run(
                query,
                action_id=action_id,
                element_id=element_id,
                order=order,
                atomic_action=atomic_action,
                action_params=action_params or "",  # Use empty string instead of None
            )
            return result.single() is not None

    def get_page_elements(self, page_id: str) -> List[Dict[str, Any]]:
        """Get all elements on a page and their executable atomic actions"""
        query = """
        MATCH (p:Page {page_id: $page_id})-[:HAS_ELEMENT]->(e:Element)
        RETURN e
        """

        with self.driver.session(database="neo4j") as session:
            result = session.run(query, page_id=page_id)
            elements = []
            for record in result:
                element = dict(record["e"])
                if "possible_actions" in element:
                    element["possible_actions"] = json.loads(
                        element["possible_actions"]
                    )
                elements.append(element)
            return elements

    def get_action_sequence(self, action_id: str) -> List[Dict[str, Any]]:
        """Get the execution sequence of a composite action"""
        query = """
        MATCH (a:Action {action_id: $action_id})-[r:COMPOSED_OF]->(e:Element)
        RETURN e.element_id as element_id, 
               e.element_type as element_type, 
               r.order as order,
               r.atomic_action as atomic_action, 
               r.action_params as action_params
        ORDER BY r.order
        """

        with self.driver.session(database="neo4j") as session:
            result = session.run(query, action_id=action_id)
            sequences = []
            for record in result:
                record_dict = {
                    "element_id": record["element_id"],
                    "element_type": record["element_type"],
                    "order": record["order"],  # Ensure this field exists
                    "atomic_action": record["atomic_action"],
                    "action_params": record["action_params"],
                }
                # Deserialize action_params
                if record_dict.get("action_params"):
                    try:
                        record_dict["action_params"] = json.loads(
                            record_dict["action_params"]
                        )
                    except json.JSONDecodeError:
                        pass  # Keep as is if not valid JSON
                sequences.append(record_dict)
            return sequences

    def add_element_leads_to(
        self,
        element_id: str,
        target_id: str,
        action_name: str,
        action_params: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.0,
    ) -> bool:
        """Create Element-LEADS_TO->Page relationship"""
        query = """
        MATCH (e:Element {element_id: $element_id})
        MATCH (t:Page {page_id: $target_id})
        MERGE (e)-[r:LEADS_TO {
            action_name: $action_name,
            action_params: $action_params,
            confidence_score: $confidence_score
        }]->(t)
        RETURN type(r) as rel_type
        """

        try:
            # Serialize action_params to JSON string
            if action_params:
                action_params = json.dumps(action_params)

            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    query,
                    element_id=element_id,
                    target_id=target_id,
                    action_name=action_name,
                    action_params=action_params or "",
                    confidence_score=confidence_score,
                )
                record = result.single()
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to create LEADS_TO relationship from element {element_id} to page {target_id}"
                    )
                return success
        except Exception as e:
            print(f"Error creating LEADS_TO relationship: {str(e)}")
            return False

    def get_chain_start_nodes(self) -> List[Dict[str, Any]]:
        """Get all chain starting nodes (Page nodes with no incoming edges)"""
        query = """
        MATCH (n:Page)
        WHERE NOT EXISTS { MATCH ()-[:LEADS_TO]->(n) }
        RETURN n
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query)
                start_nodes = []
                for record in result:
                    node = dict(record["n"])
                    start_nodes.append(node)
                return start_nodes
        except Exception as e:
            print(f"Error getting chain start nodes: {str(e)}")
            return []

    def get_chain_from_start(self, start_page_id: str) -> List[List[Dict[str, Any]]]:
        """Get complete operation chain from starting node, returning triplet chain structure

        Args:
            start_page_id: ID of the starting page

        Returns:
            List[List[Dict]]: List containing complete chain information, each chain consists of multiple triplets
            Each triplet contains:
                - source_page: Source page node information
                - element: Element node information
                - target_page: Target page node information
                - action: Action information
        """
        query = """
        MATCH path = (start:Page {page_id: $start_page_id})-[:HAS_ELEMENT|LEADS_TO*]->(end:Page)
        WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() }  // Ensure it's an endpoint page
        WITH path, relationships(path) as rels, nodes(path) as nodes
        WITH DISTINCT [n in nodes | n{.*}] as node_props,
             [r in rels | r{.*}] as rel_props
        RETURN node_props, rel_props
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, start_page_id=start_page_id)
                chains = []
                seen_chains = set()  # For deduplication

                for record in result:
                    nodes = record["node_props"]
                    rels = record["rel_props"]

                    # Build triplet chain
                    chain = []
                    current_page = nodes[0]
                    i = 0

                    while i < len(rels):
                        # Handle HAS_ELEMENT relationship
                        if "element_id" in nodes[i + 1]:  # Found element node
                            element = nodes[i + 1]
                            # Continue looking for LEADS_TO relationship
                            if i + 1 < len(rels) and "action_name" in rels[i + 1]:
                                target_page = nodes[i + 2]
                                # Build triplet
                                triplet = {
                                    "source_page": current_page,
                                    "element": element,
                                    "target_page": target_page,
                                    "action": rels[i + 1],
                                }
                                chain.append(triplet)
                                current_page = target_page  # Update current page
                                i += 2  # Skip two processed relationships
                            else:
                                i += 1
                        else:
                            i += 1

                    if chain:  # Only add non-empty chains
                        # Create unique identifier for the chain
                        chain_key = tuple(
                            (
                                t["source_page"]["page_id"],
                                t["element"]["element_id"],
                                t["target_page"]["page_id"],
                            )
                            for t in chain
                        )
                        if chain_key not in seen_chains:
                            seen_chains.add(chain_key)
                            chains.append(chain)

                return chains[0]
        except Exception as e:
            print(f"Error getting chain from start node: {str(e)}")
            return []

    def get_all_actions(self) -> List[Dict[str, Any]]:
        """Get all Action nodes from the database

        Returns:
            List[Dict[str, Any]]: List containing all Action node information
        """
        query = """
        MATCH (a:Action)
        RETURN a
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    actions.append(action)
                return actions
        except Exception as e:
            print(f"Error getting all actions: {str(e)}")
            return []

    def get_action_by_id(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get Action node by ID

        Args:
            action_id: ID of the Action node

        Returns:
            Dict[str, Any] or None: Action node information, or None if not found
        """
        query = """
        MATCH (a:Action)
        WHERE a.action_id = $action_id
        RETURN a
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, action_id=action_id)
                record = result.single()
                if record:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    return action
                return None
        except Exception as e:
            print(f"Error getting action by ID {action_id}: {str(e)}")
            return None

    def get_element_by_id(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get Element node by ID

        Args:
            element_id: ID of the Element node

        Returns:
            Dict[str, Any] or None: Element node information, or None if not found
        """
        query = """
        MATCH (e:Element)
        WHERE e.element_id = $element_id
        RETURN e
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    element = dict(record["e"])
                    # Deserialize JSON string fields, if any
                    for field in ["possible_actions", "other_info"]:
                        if field in element and isinstance(element[field], str):
                            try:
                                element[field] = json.loads(element[field])
                            except json.JSONDecodeError:
                                pass  # Keep as is if not valid JSON
                    return element
                return None
        except Exception as e:
            print(f"Error getting element by ID {element_id}: {str(e)}")
            return None

    def get_all_high_level_actions(self) -> List[Dict[str, Any]]:
        """Get all high-level action nodes from the database

        Returns:
            List[Dict[str, Any]]: List containing all high-level action node information
        """
        query = """
        MATCH (a:Action)
        WHERE a.is_high_level = true OR a.high_level = true OR a.type = 'high_level'
        RETURN a
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    actions.append(action)
                return actions
        except Exception as e:
            print(f"Error getting all high level actions: {str(e)}")
            return []

    def get_high_level_actions_for_task(self, task: str) -> List[Dict[str, Any]]:
        """Get high-level action nodes related to a specific task

        Args:
            task: User task description

        Returns:
            List[Dict[str, Any]]: List containing relevant high-level action node information
        """
        # First try fuzzy matching by task description
        query = """
        MATCH (a:Action)
        WHERE (a.is_high_level = true OR a.high_level = true OR a.type = 'high_level')
        AND (a.name CONTAINS $task OR a.description CONTAINS $task)
        RETURN a
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, task=task)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    # Deserialize JSON string fields
                    if "element_sequence" in action and isinstance(
                        action["element_sequence"], str
                    ):
                        try:
                            action["element_sequence"] = json.loads(
                                action["element_sequence"]
                            )
                        except json.JSONDecodeError:
                            pass  # Keep as is if not valid JSON
                    actions.append(action)

                # If no results found, return all high-level actions for further processing
                if not actions:
                    return self.get_all_high_level_actions()

                return actions
        except Exception as e:
            print(f"Error getting high level actions for task '{task}': {str(e)}")
            return []

    def get_shortcuts_for_action(self, action_id: str) -> List[Dict[str, Any]]:
        """Get shortcuts associated with a specific high-level action

        Args:
            action_id: High-level action ID

        Returns:
            List[Dict[str, Any]]: List containing associated shortcut information
        """
        query = """
        MATCH (s:Shortcut)-[:REFERS_TO]->(a:Action {action_id: $action_id})
        RETURN s
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, action_id=action_id)
                shortcuts = []
                for record in result:
                    shortcut = dict(record["s"])
                    # Deserialize JSON string fields
                    for field in ["conditions", "page_flow"]:
                        if field in shortcut and isinstance(shortcut[field], str):
                            try:
                                shortcut[field] = json.loads(shortcut[field])
                            except json.JSONDecodeError:
                                pass  # Keep as is if not valid JSON
                    shortcuts.append(shortcut)
                return shortcuts
        except Exception as e:
            print(f"Error getting shortcuts for action '{action_id}': {str(e)}")
            return []

    def get_page_by_visual_embedding(
        self, embedding_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get page node by visual embedding ID

        Args:
            embedding_id: Visual embedding ID

        Returns:
            Dict[str, Any] or None: Page node information, or None if not found
        """
        query = """
        MATCH (p:Page)
        WHERE p.visual_embedding_id = $embedding_id
        RETURN p
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, embedding_id=embedding_id)
                record = result.single()
                if record:
                    page = dict(record["p"])
                    # Handle possible JSON fields
                    for field in ["elements_data", "metadata"]:
                        if field in page and isinstance(page[field], str):
                            try:
                                page[field] = json.loads(page[field])
                            except json.JSONDecodeError:
                                pass  # Keep as is if not valid JSON
                    return page
                return None
        except Exception as e:
            print(f"Error getting page by visual embedding ID {embedding_id}: {str(e)}")
            return None
