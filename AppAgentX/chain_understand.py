from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, SecretStr
import json
import os
from langchain_openai import ChatOpenAI
from .data.graph_db import Neo4jDatabase
from . import config # This will import AppAgentX/config.py

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "LearnTriplet"  # Keep specific project name

model = ChatOpenAI(
    openai_api_base=config.LLM_BASE_URL,
    openai_api_key=config.LLM_API_KEY,
    model_name=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=config.LLM_MAX_TOKEN,
)

URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db = Neo4jDatabase(URI, AUTH)


# Model definition for reasoning results
class TripletReasoning(BaseModel):
    context: str = Field(description="Context description of the operation")
    user_intent: str = Field(description="User intention analysis")
    state_change: str = Field(description="State change description")
    task_relation: str = Field(description="Relationship with the task")
    source_page_enhanced_desc: str = Field(
        description="Enhanced description of the source page"
    )
    element_enhanced_desc: str = Field(
        description="Enhanced description of the element"
    )
    target_page_enhanced_desc: str = Field(
        description="Enhanced description of the target page"
    )


# Build LCEL chain for triplet reasoning
def create_triplet_reasoning_chain():
    """Create LCEL chain for triplet reasoning"""
    # Define reasoning prompt template
    triplet_reasoning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in understanding and reasoning about UI operation chains. You need to analyze the given page-element-page triplet information and perform deep understanding and reasoning. You will receive textual descriptions and screenshots of pages, please analyze both.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": """Please analyze the following UI operation triplet information:
        Source Page: {source_page_desc}
        Element: {element_desc}
        Target Page: {target_page_desc}
        Action: {action_name}
        
        Please reason and expand from the following aspects:
        1. What is the context and purpose of this operation?
        2. What might be the user's intention when performing this operation?
        3. How does this operation affect the page state change?
        4. What is the relationship between this operation and the overall task flow?
        5. Based on your understanding, generate richer and more accurate descriptions for the source page, element, and target page.
        
        Please return your reasoning results in a structured way, including the following fields:
        - context: Operation context description
        - user_intent: User intention analysis
        - state_change: State change description
        - task_relation: Relationship with the task
        - source_page_enhanced_desc: Enhanced description of source page
        - element_enhanced_desc: Enhanced description of element
        - target_page_enhanced_desc: Enhanced description of target page
        
        {format_instructions}""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{source_page_image}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{target_page_image}"},
                    },
                ],
            ),
        ]
    )

    # Use JsonOutputParser instead of StrOutputParser
    parser = JsonOutputParser(pydantic_object=TripletReasoning)

    # Inject format instructions into prompt template
    prompt = triplet_reasoning_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # Build LCEL chain
    reasoning_chain = RunnablePassthrough() | prompt | model | parser

    return reasoning_chain


# Build LCEL chain for description merging
def create_merge_descriptions_chain():
    """Create LCEL chain for merging page descriptions

    Returns:
        Description merging chain
    """
    # Define merge description prompt template
    merge_descriptions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in merging and optimizing page descriptions. You need to analyze descriptions of shared pages between two adjacent triplets and merge them into a more complete description.",
            ),
            (
                "human",
                """Please analyze the following two descriptions that describe the same page but from different contexts:

        Current Task: {task_info}
        
        Description 1 (as target page of previous triplet): {desc1}
        Description 2 (as source page of next triplet): {desc2}
        
        Please merge these two descriptions to generate a more complete and coherent description, requirements:
        1. Consider the context and goals of the current task
        2. Preserve all important information
        3. Eliminate redundant content
        4. Ensure description coherence
        5. Highlight core functionality and features of the page
        6. Emphasize relevance to the current task
        
        Please return the merged description.""",
            ),
        ]
    )

    # Build LCEL chain
    merge_chain = (
        RunnablePassthrough() | merge_descriptions_prompt | model | StrOutputParser()
    )

    return merge_chain


# Process single triplet
async def process_triplet(triplet: Dict[str, Any], reasoning_chain):
    """Process reasoning for a single triplet

    Args:
        triplet: Triplet containing source page, element, target page and action information
        reasoning_chain: Triplet reasoning chain

    Returns:
        Triplet with added reasoning results
    """
    # Prepare reasoning input
    reasoning_input = {
        "source_page_desc": triplet["source_page"].get("description", ""),
        "element_desc": triplet["element"].get("description", ""),
        "target_page_desc": triplet["target_page"].get("description", ""),
        "action_name": triplet["action"].get("action_name", ""),
    }

    # Load source and target page images
    try:
        from utils import load_image_to_base64

        # Load source page image
        source_page_image = "data:image/png;base64,"
        if (
            "raw_page_url" in triplet["source_page"]
            and triplet["source_page"]["raw_page_url"]
        ):
            source_image_path = triplet["source_page"]["raw_page_url"]
            source_page_image = load_image_to_base64(source_image_path)

        # Load target page image
        target_page_image = "data:image/png;base64,"
        if (
            "raw_page_url" in triplet["target_page"]
            and triplet["target_page"]["raw_page_url"]
        ):
            target_image_path = triplet["target_page"]["raw_page_url"]
            target_page_image = load_image_to_base64(target_image_path)

        # Add images to reasoning input
        reasoning_input["source_page_image"] = source_page_image
        reasoning_input["target_page_image"] = target_page_image
    except Exception as e:
        print(f"Error loading page images: {str(e)}")
        # Use empty base64 image if loading fails
        reasoning_input["source_page_image"] = "data:image/png;base64,"
        reasoning_input["target_page_image"] = "data:image/png;base64,"
    try:
        # Execute reasoning - now returns dictionary object
        reasoning_result = await reasoning_chain.ainvoke(reasoning_input)

        # Store result directly as reasoning field
        triplet["reasoning"] = reasoning_result

        # Update descriptions in triplet - update enhanced descriptions directly to description field
        triplet["source_page"]["description"] = reasoning_result[
            "source_page_enhanced_desc"
        ]
        triplet["element"]["description"] = reasoning_result["element_enhanced_desc"]
        triplet["target_page"]["description"] = reasoning_result[
            "target_page_enhanced_desc"
        ]

    except Exception as e:
        print(f"Error during triplet reasoning: {str(e)}")
        # Record detailed error information for debugging
        triplet["reasoning_error"] = str(e)

    return triplet


# Merge descriptions of shared nodes in adjacent triplets
async def merge_node_descriptions(
    chain: List[Dict[str, Any]], merge_chain, task_info: str
):
    """Merge description information of overlapping nodes in the chain

    Args:
        chain: Triplet chain
        merge_chain: Description merging chain
        task_info: Task information

    Returns:
        Triplet chain with updated descriptions
    """
    for i in range(len(chain) - 1):
        current_triplet = chain[i]
        next_triplet = chain[i + 1]

        # Check for overlapping nodes
        if (
            current_triplet["target_page"]["page_id"]
            == next_triplet["source_page"]["page_id"]
        ):
            # Prepare merge input
            merge_input = {
                "desc1": current_triplet["target_page"].get("description", ""),
                "desc2": next_triplet["source_page"].get("description", ""),
                "task_info": task_info,  # Add task information
            }

            try:
                # Execute description merge
                merged_desc = await merge_chain.ainvoke(merge_input)

                # Update descriptions in both triplets
                current_triplet["target_page"]["description"] = merged_desc
                next_triplet["source_page"]["description"] = merged_desc

                # Update node description in database
                # TODO: Implement database node description update
                # Can use neo4j_db.update_node_property method to update node properties
            except Exception as e:
                print(f"Error merging descriptions: {str(e)}")

    return chain


# Update node properties in database
def update_node_in_db(
    node_id: str,
    property_name: str,
    property_value: Any,
    node_type: Optional[str] = None,
) -> bool:
    """Update node properties in database

    Args:
        node_id: Node ID
        property_name: Property name
        property_value: Property value
        node_type: Node type (optional)

    Returns:
        Whether update was successful
    """
    try:
        return db.update_node_property(
            node_id=node_id,
            property_name=property_name,
            property_value=property_value,
            node_type=node_type,
        )
    except Exception as e:
        print(f"Error updating node property: {str(e)}")
        return False


# Process single chain
async def process_single_chain(
    chain: List[Dict[str, Any]], reasoning_chain, merge_chain
) -> List[Dict[str, Any]]:
    """Process all triplets in a single chain and merge descriptions

    Args:
        chain: Single triplet chain
        reasoning_chain: Triplet reasoning chain
        merge_chain: Description merging chain

    Returns:
        Processed chain
    """
    # Extract task information from first node in chain
    task_info = "Unknown Task"
    if (
        chain
        and chain[0]
        and "source_page" in chain[0]
        and "other_info" in chain[0]["source_page"]
    ):
        try:
            other_info = chain[0]["source_page"]["other_info"]
            if isinstance(other_info, str):
                other_info = json.loads(other_info)

            if "task_info" in other_info and "description" in other_info["task_info"]:
                task_info = other_info["task_info"]["description"]
                print(f"Extracted task information: {task_info}")
        except Exception as e:
            print(f"Error extracting task information: {str(e)}")

    # Process each triplet
    processed_triplets = []
    for triplet in chain:
        processed_triplet = await process_triplet(triplet, reasoning_chain)
        processed_triplets.append(processed_triplet)

    # Merge node descriptions, pass task information
    merged_chain = await merge_node_descriptions(
        processed_triplets, merge_chain, task_info
    )

    # Update node information in database
    for triplet in merged_chain:
        # Update source page description
        update_node_in_db(
            triplet["source_page"]["page_id"],
            "description",
            triplet["source_page"].get("description", ""),
            "Page",
        )

        # Update target page description
        update_node_in_db(
            triplet["target_page"]["page_id"],
            "description",
            triplet["target_page"].get("description", ""),
            "Page",
        )

        # Update element description
        update_node_in_db(
            triplet["element"]["element_id"],
            "description",
            triplet["element"].get("description", ""),
            "Element",
        )

        # Save reasoning results (if exists)
        if "reasoning" in triplet:
            # Save complete reasoning results to element node
            update_node_in_db(
                triplet["element"]["element_id"],
                "reasoning",
                json.dumps(triplet["reasoning"]),
                "Element",
            )

    return merged_chain


# Process complete chain and update database
async def process_and_update_chain(start_page_id: str) -> List[Dict[str, Any]]:
    """Process triplet chain and update database

    Args:
        start_page_id: Starting page ID

    Returns:
        List of processed triplets
    """
    # Create necessary reasoning chains
    reasoning_chain = create_triplet_reasoning_chain()
    merge_chain = create_merge_descriptions_chain()

    # Extract chain data
    triplets = db.get_chain_from_start(
        start_page_id
    )  # Extract chain triplet data from database

    if not triplets:
        print(f"Warning: No triplets found")
        return []

    # Directly process returned triplet list
    processed_chain = await process_single_chain(triplets, reasoning_chain, merge_chain)

    return processed_chain
