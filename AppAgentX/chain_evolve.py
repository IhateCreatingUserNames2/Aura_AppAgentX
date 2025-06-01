import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from . import config # This will import AppAgentX/config.py
from .data.graph_db import Neo4jDatabase


# Configure environment variables
os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "ChainEvolve"

# Initialize LLM model
model = ChatOpenAI(
    openai_api_base=config.LLM_BASE_URL,
    openai_api_key=config.LLM_API_KEY,
    model_name=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=2000,
)

# Initialize database connection
URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db = Neo4jDatabase(URI, AUTH)


# Chain evaluation result model
class ChainEvaluationResult(BaseModel):
    is_templateable: bool = Field(description="Whether the chain can be templated")
    confidence_score: float = Field(
        description="Confidence score for templateability (0-1)"
    )
    reason: str = Field(description="Reason and explanation for the evaluation")
    suggested_name: str = Field(description="Suggested name for the high-level action")


# High-level node generation result model
class ActionNodeGeneration(BaseModel):
    action_id: str = Field(description="High-level action node ID")
    name: str = Field(description="High-level action name")
    description: str = Field(description="Detailed description")
    preconditions: List[str] = Field(
        description="Preconditions for executing the high-level action"
    )
    element_sequence: List[Dict[str, Any]] = Field(
        description="Sequence of elements included in the high-level action"
    )
    template_pattern: Dict[str, Any] = Field(description="Template matching pattern")


# Create chain evaluation chain
def create_chain_evaluation_chain():
    """Create an LCEL chain for evaluating whether the chain can be templated."""
    # Define evaluation prompt template
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in evaluating whether UI operation chains can be templated. You need to analyze the given UI operation chain and determine if it has the potential for templating.",
            ),
            (
                "human",
                """Please evaluate whether the following UI operation chain can be templated into a high-level action:

Task description: {task_description}

Chain operations:
{chain_operations}

Please evaluate from the following aspects:
1. Does this operation chain have clear start and end steps?
2. Do the operations in the chain have clear business logic and goals?
3. Do these operations form a complete and meaningful task flow?
4. Is it possible to reuse this chain in other similar tasks?
5. Are there obvious parameterizable parts?

Please return your evaluation results in a structured manner, including the following fields:
- is_templateable: Whether it can be templated (boolean)
- confidence_score: Confidence score (float between 0-1)
- reason: Detailed evaluation reason
- suggested_name: If it can be templated, the suggested high-level action name

{format_instructions}""",
            ),
        ]
    )

    # Use JsonOutputParser
    parser = JsonOutputParser(pydantic_object=ChainEvaluationResult)

    # Inject format instructions into the prompt template
    prompt = evaluation_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # Build LCEL chain
    evaluation_chain = RunnablePassthrough() | prompt | model | parser

    return evaluation_chain


# Create action generation chain
def create_action_generation_chain():
    """Create an LCEL chain for generating high-level action node content."""
    # Define generation prompt template
    generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specialized in generating high-level UI operation nodes. You need to generate a complete description of a high-level action node based on the given chain information.",
            ),
            (
                "human",
                """Please generate a high-level action node based on the following UI operation chain information:

Task description: {task_description}

Chain operations:
{chain_operations}

Chain element details:
{element_details}

Chain reasoning results:
{reasoning_results}

Please generate a complete description of the high-level action node, including the following fields:
- action_id: Generate a unique ID for the high-level action (format like: "high_level_action_xxx")
- name: Concise name of the high-level action
- description: Detailed description of the function, purpose, and execution process of the high-level action
- preconditions: List of preconditions for executing the high-level action
- element_sequence: Sequence of elements included in the high-level action, each element contains:
  * element_id: Element ID
  * order: Order of operation
  * atomic_action: Atomic action performed on the element
  * action_params: Action parameters (if any)
- template_pattern: Template matching pattern, including:
  * criteria: Applicable matching conditions
  * parameter_fields: Parameterizable fields and their descriptions

{format_instructions}""",
            ),
        ]
    )

    # Use JsonOutputParser
    parser = JsonOutputParser(pydantic_object=ActionNodeGeneration)

    # Inject format instructions into the prompt template
    prompt = generation_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # Build LCEL chain
    generation_chain = RunnablePassthrough() | prompt | model | parser

    return generation_chain


# Extract task description
def extract_task_description(chain: List[Dict[str, Any]]) -> str:
    """Extract task description from the chain.

    Args:
        chain: Triplet chain

    Returns:
        Extracted task description
    """
    task_info = "Unknown task"
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
        except Exception as e:
            print(f"Error extracting task information: {str(e)}")

    return task_info


# Format chain operations as text description
def format_chain_operations(chain: List[Dict[str, Any]]) -> str:
    """Format chain operations as text description.

    Args:
        chain: Triplet chain

    Returns:
        Formatted operation description text
    """
    operations = []

    for i, triplet in enumerate(chain):
        source_page = triplet["source_page"].get("description", "Unknown page")
        element = triplet["element"].get("description", "Unknown element")
        target_page = triplet["target_page"].get("description", "Unknown page")
        action_name = triplet["action"].get("action_name", "Unknown operation")

        operation = f"Step {i+1}: On the page 【{source_page}】, perform the operation 【{action_name}】 on 【{element}】 to reach the page 【{target_page}】."
        operations.append(operation)

    return "\n".join(operations)


# Extract element details
def extract_element_details(chain: List[Dict[str, Any]]) -> str:
    """Extract detailed information of all elements in the chain.

    Args:
        chain: Triplet chain

    Returns:
        Element detail text
    """
    elements = []

    for i, triplet in enumerate(chain):
        element_id = triplet["element"].get("element_id", "Unknown ID")
        element_type = triplet["element"].get("element_type", "Unknown type")
        element_desc = triplet["element"].get("description", "Unknown description")
        action_name = triplet["action"].get("action_name", "Unknown operation")

        element_detail = f"Element {i+1}:\n  ID: {element_id}\n  Type: {element_type}\n  Description: {element_desc}\n  Related operation: {action_name}"
        elements.append(element_detail)

    return "\n".join(elements)


# Extract reasoning results
def extract_reasoning_results(chain: List[Dict[str, Any]]) -> str:
    """Extract reasoning results of all triplets in the chain.

    Args:
        chain: Triplet chain

    Returns:
        Reasoning result text
    """
    reasoning_texts = []

    for i, triplet in enumerate(chain):
        if "reasoning" in triplet:
            reasoning = triplet["reasoning"]

            reasoning_text = f"Step {i+1} reasoning result:\n"
            reasoning_text += f"  Context: {reasoning.get('context', 'N/A')}\n"
            reasoning_text += f"  User intent: {reasoning.get('user_intent', 'N/A')}\n"
            reasoning_text += (
                f"  State change: {reasoning.get('state_change', 'N/A')}\n"
            )
            reasoning_text += (
                f"  Task relation: {reasoning.get('task_relation', 'N/A')}\n"
            )

            reasoning_texts.append(reasoning_text)

    return (
        "\n".join(reasoning_texts)
        if reasoning_texts
        else "No available reasoning results"
    )


# Evaluate whether the chain can be templated
async def evaluate_chain_templateability(
    chain: List[Dict[str, Any]]
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Evaluate whether the chain can be templated into a high-level action.

    Args:
        chain: Triplet chain

    Returns:
        (Whether it can be templated, evaluation result dictionary)
    """
    # Create evaluation chain
    evaluation_chain = create_chain_evaluation_chain()

    # Prepare evaluation input
    task_description = extract_task_description(chain)
    chain_operations = format_chain_operations(chain)

    evaluation_input = {
        "task_description": task_description,
        "chain_operations": chain_operations,
    }

    try:
        # Execute evaluation - note that this returns a dictionary rather than a Pydantic object
        evaluation_result = await evaluation_chain.ainvoke(evaluation_input)

        # Check if the returned result contains the required fields
        if (
            isinstance(evaluation_result, dict)
            and "is_templateable" in evaluation_result
        ):
            is_templateable = evaluation_result["is_templateable"]
            return is_templateable, evaluation_result
        else:
            print(
                f"Warning: The format of the evaluation result returned by LLM is incorrect: {evaluation_result}"
            )
            return False, None
    except Exception as e:
        print(f"Error evaluating the chain: {str(e)}")
        return False, None


# Generate high-level action node
async def generate_action_node(chain: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Generate high-level action node content.

    Args:
        chain: Triplet chain

    Returns:
        Generated high-level action node content (dictionary)
    """
    # Create generation chain
    generation_chain = create_action_generation_chain()

    # Prepare generation input
    task_description = extract_task_description(chain)
    chain_operations = format_chain_operations(chain)
    element_details = extract_element_details(chain)
    reasoning_results = extract_reasoning_results(chain)

    generation_input = {
        "task_description": task_description,
        "chain_operations": chain_operations,
        "element_details": element_details,
        "reasoning_results": reasoning_results,
    }

    try:
        # Execute generation - note that this returns a dictionary rather than a Pydantic object
        generation_result = await generation_chain.ainvoke(generation_input)

        # Check if the returned result is a valid dictionary
        if isinstance(generation_result, dict) and "action_id" in generation_result:
            return generation_result
        else:
            print(
                f"Warning: The format of the generation result returned by LLM is incorrect: {generation_result}"
            )
            return None
    except Exception as e:
        print(f"Error generating high-level action node: {str(e)}")
        return None


# Create high-level action node in the database
def create_action_node_in_db(action_data: Dict[str, Any]) -> Optional[str]:
    """Create a high-level action node in the database.

    Args:
        action_data: High-level action node data dictionary

    Returns:
        Created node ID or None
    """
    try:
        # Prepare node properties
        properties = {
            "action_id": action_data["action_id"],
            "name": action_data["name"],
            "description": action_data["description"],
            "preconditions": json.dumps(action_data["preconditions"]),
            "element_sequence": action_data[
                "element_sequence"
            ],  # Will be automatically serialized in graph_db
            "template_pattern": json.dumps(action_data["template_pattern"]),
            "is_high_level": True,  # Mark as high-level action
        }

        # Create node
        node_id = db.create_action(properties)

        if not node_id:
            print("Failed to create high-level action node")
            return None

        print(f"Successfully created high-level action node, ID: {node_id}")
        return node_id
    except Exception as e:
        print(f"Error creating high-level action node: {str(e)}")
        return None


# Create relations between high-level action and elements
def create_action_element_relations(action_data: Dict[str, Any]) -> bool:
    """Create relations between high-level action and elements.

    Args:
        action_data: High-level action node data dictionary

    Returns:
        Whether all succeeded
    """
    success = True

    try:
        # Iterate through element sequence to create relations
        for element_info in action_data["element_sequence"]:
            element_id = element_info.get("element_id")
            order = element_info.get("order")
            atomic_action = element_info.get("atomic_action")
            action_params = element_info.get("action_params", {})

            # Create relation
            relation_success = db.add_element_to_action(
                action_id=action_data["action_id"],
                element_id=element_id,
                order=order,
                atomic_action=atomic_action,
                action_params=action_params,
            )

            if not relation_success:
                print(
                    f"Failed to create Action-Element relation, Element ID: {element_id}"
                )
                success = False
    except Exception as e:
        print(f"Error creating action-element relations: {str(e)}")
        success = False

    return success


# Main processing function
async def evolve_chain_to_action(start_page_id: str) -> Optional[str]:
    """Process the chain evolution into a high-level action node.

    Args:
        start_page_id: Starting page ID

    Returns:
        Created high-level action node ID or None
    """
    try:
        # 1. Get the complete chain
        print(f"Getting the chain starting from page {start_page_id}...")
        chain = db.get_chain_from_start(start_page_id)

        if not chain:
            print(f"No chain found starting from {start_page_id}")
            return None

        print(f"Successfully retrieved the chain, total {len(chain)} triplets")

        # 2. Evaluate whether the chain can be templated
        print("Evaluating whether the chain can be templated...")
        is_templateable, evaluation_result = await evaluate_chain_templateability(chain)

        if not is_templateable:
            reason = (
                "No reason provided"
                if evaluation_result is None
                else evaluation_result.get("reason", "No reason provided")
            )
            print(f"The chain is evaluated as non-templatable: {reason}")
            return None

        print(
            f"The chain is evaluated as templatable, confidence: {evaluation_result.get('confidence_score', 0):.2f}"
        )
        print(f"Suggested name: {evaluation_result.get('suggested_name', 'Unnamed')}")
        print(f"Reason: {evaluation_result.get('reason', 'No reason provided')}")

        # 3. Generate high-level action node content
        print("Generating high-level action node content...")
        action_data = await generate_action_node(chain)

        if not action_data:
            print("Failed to generate high-level action node content")
            return None

        print(
            f"Successfully generated high-level action node content: {action_data['name']}"
        )

        # 4. Create high-level action node
        print("Creating high-level action node in the database...")
        node_id = create_action_node_in_db(action_data)

        if not node_id:
            print("Failed to create high-level action node")
            return None

        # 5. Create relations between high-level action and elements
        print("Creating relations between high-level action and elements...")
        relations_success = create_action_element_relations(action_data)

        if not relations_success:
            print("Some element relations creation failed")

        print(
            f"Successfully completed chain evolution, created high-level action node: {action_data['name']} (ID: {action_data['action_id']})"
        )
        return action_data["action_id"]
    except Exception as e:
        print(f"Error processing chain evolution: {str(e)}")
        return None


# Test function
async def run_test():
    """Run tests."""
    try:
        print("\n===== Chain Evolution Test =====")

        # 1. Get starting nodes
        print("\n1. Getting starting nodes...")
        start_nodes = db.get_chain_start_nodes()

        if not start_nodes:
            print("❌ No starting nodes found, test terminated")
            return

        start_page_id = start_nodes[0]["page_id"]
        print(f"✓ Using starting page ID: {start_page_id}")

        # 2. Execute chain evolution
        print("\n2. Starting chain evolution...")
        action_id = await evolve_chain_to_action(start_page_id)

        if not action_id:
            print("\n❌ Chain evolution failed")
            return

        print(f"\n✓ Chain evolution succeeded! High-level action node ID: {action_id}")

        print("\n===== Test Completed ✨ =====")
    except Exception as e:
        print(f"\n❌ Test error: {str(e)}")


# Run tests when this file is executed directly
if __name__ == "__main__":
    # Run asynchronous test function
    asyncio.run(run_test())
