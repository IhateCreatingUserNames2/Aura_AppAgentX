import base64
from io import BytesIO
from typing import Union
from PIL import Image
from .data.graph_db import Neo4jDatabase
from .data.vector_db import VectorStore, NodeType
from typing import Optional
from . import config # This will import AppAgentX/config.py


def load_image_to_base64(image_input: Union[str, Image.Image, BytesIO]) -> str:
    """
    Convert image to base64 encoding

    Args:
        image_input: Image path string, PIL Image object, or BytesIO object

    Returns:
        str: base64 encoded image string
    """
    try:
        if isinstance(image_input, str):
            with Image.open(image_input) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
        elif isinstance(image_input, Image.Image):
            buffered = BytesIO()
            image_input.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        elif isinstance(image_input, BytesIO):
            # Save current position
            current_pos = image_input.tell()
            # Move pointer to start
            image_input.seek(0)
            # Read data and encode
            img_str = base64.b64encode(image_input.getvalue()).decode()
            # Restore pointer position
            image_input.seek(current_pos)
            return f"data:image/png;base64,{img_str}"
        else:
            raise ValueError(
                "Input must be an image path string, PIL Image object, or BytesIO object"
            )
    except Exception as e:
        print(f"Failed to convert image: {str(e)}")
        return ""


def clear_neo4j_database(uri: str, auth: tuple) -> bool:
    """
    Clear all nodes and relationships in the Neo4j graph database

    Args:
        uri: Neo4j database connection URI
        auth: Authentication information tuple (username, password)

    Returns:
        bool: Whether the operation was successful
    """
    try:
        db = Neo4jDatabase(uri, auth)

        # Use Cypher query to delete all relationships and nodes
        with db.driver.session(database="neo4j") as session:
            # First delete all relationships
            session.run("MATCH ()-[r]-() DELETE r")
            # Then delete all nodes
            session.run("MATCH (n) DELETE n")

        print("✓ Successfully cleared Neo4j database")
        return True

    except Exception as e:
        print(f"❌ Failed to clear Neo4j database: {str(e)}")
        return False
    finally:
        if "db" in locals():
            db.close()


def clear_vector_store(
    api_key: str, index_name: str = "area-embedding", namespace: Optional[str] = None
) -> bool:
    """
    Clear vectors in the Pinecone vector database

    Args:
        api_key: Pinecone API key
        index_name: Index name
        namespace: Optional, specify the namespace to clear. If None, clear all namespaces

    Returns:
        bool: Whether the operation was successful
    """
    try:
        vector_store = VectorStore(api_key=api_key, index_name=index_name)

        if namespace:
            # Clear specified namespace
            try:
                vector_store.index.delete(delete_all=True, namespace=namespace)
                print(f"✓ Successfully cleared vector database namespace: {namespace}")
            except Exception as e:
                if "Namespace not found" in str(e):
                    print(
                        f"ℹ️ Namespace {namespace} does not exist or has already been cleared"
                    )
                else:
                    print(f"❌ Failed to clear namespace {namespace}: {str(e)}")
                return False
        else:
            # Clear all namespaces
            for node_type in NodeType:
                try:
                    vector_store.index.delete(
                        delete_all=True, namespace=node_type.value
                    )
                    print(f"✓ Successfully cleared namespace: {node_type.value}")
                except Exception as e:
                    if "Namespace not found" in str(e):
                        print(
                            f"ℹ️ Namespace {node_type.value} does not exist or has already been cleared"
                        )
                    else:
                        print(
                            f"❌ Failed to clear namespace {node_type.value}: {str(e)}"
                        )
                    continue
            print("✓ Completed clearing all namespaces")

        return True

    except Exception as e:
        print(f"❌ Failed to clear vector database: {str(e)}")
        return False


def clear_all_databases(
    neo4j_uri: str,
    neo4j_auth: tuple,
    pinecone_api_key: str,
    pinecone_index_name: str = "area-embedding",
) -> bool:
    """
    Clear all databases (Neo4j and Pinecone)

    Args:
        neo4j_uri: Neo4j database connection URI
        neo4j_auth: Neo4j authentication information tuple (username, password)
        pinecone_api_key: Pinecone API key
        pinecone_index_name: Pinecone index name

    Returns:
        bool: Whether all operations were successful
    """
    print("\nStarting to clear databases...")

    # Clear Neo4j
    neo4j_success = clear_neo4j_database(neo4j_uri, neo4j_auth)

    # Clear Pinecone
    vector_success = clear_vector_store(pinecone_api_key, pinecone_index_name)

    # Print summary
    if neo4j_success and vector_success:
        print("\n✨ All databases cleared successfully!")
    else:
        print("\n⚠️ Some databases failed to clear, please check error messages")

    return neo4j_success and vector_success


if __name__ == "__main__":
    # Use configurations from the config file
    NEO4J_URI = config.Neo4j_URI
    NEO4J_AUTH = config.Neo4j_AUTH
    PINECONE_API_KEY = config.PINECONE_API_KEY

    # Execute clear operation
    clear_all_databases(
        neo4j_uri=NEO4J_URI, neo4j_auth=NEO4J_AUTH, pinecone_api_key=PINECONE_API_KEY
    )
