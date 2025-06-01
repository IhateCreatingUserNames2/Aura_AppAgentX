import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from uuid import uuid4

from .. import config
from .State import State
from .graph_db import Neo4jDatabase
from .vector_db import VectorStore, VectorData, NodeType
from ..tool.img_tool import element_img, extract_features


def generate_short_md5(input_string, length=8):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode("utf-8"))
    full_md5 = md5_hash.hexdigest()
    short_md5 = full_md5[:length]
    return short_md5


def state2json(state: State, save_path: str = None) -> str:
    """
    Export the running process to JSON
    """
    if not isinstance(state, dict):
        raise TypeError("The 'state' parameter must be a dictionary.")

    # Create a new dictionary to store the fields to be saved
    filtered_state = {
        "tsk": state.get("tsk", ""),
        "history_steps": state.get("history_steps", []),
        "app_name": state.get("app_name", ""),
        "step": state.get("step", 0),
        # Add final_page field, including the screenshot and JSON information of the last page
        "final_page": {
            "screenshot": state.get("current_page_screenshot", ""),
            "page_json": (
                state.get("current_page_json", {}).get("parsed_content_json_path", "")
                if isinstance(state.get("current_page_json"), dict)
                else state.get("current_page_json", "")
            ),
        },
    }

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./log/json_state/state_{timestamp}.json"

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_state, f, ensure_ascii=False, indent=4)
        return f"State has been successfully saved to {output_path.resolve()}"
    except Exception as e:
        return f"Error saving state to JSON: {e}"


def pos2id(x: int, y: int, json_path: str) -> Optional[Dict]:
    """
    Find matching element information from the JSON file based on coordinates.
    If no direct match is found, return the nearest element.

    Args:
        x: The x-coordinate of the click position.
        y: The y-coordinate of the click position.
        json_path: The path to the element JSON file.

    Returns:
        Optional[Dict]: A dictionary of the matched element information, or None if not found.
    """
    try:
        # Read the element JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            elements_data = json.load(f)

        # Convert absolute coordinates to relative coordinates
        screen_width = 1080  # Assume screen width
        screen_height = 1920  # Assume screen height
        norm_x = x / screen_width
        norm_y = y / screen_height

        # Match specific elements from element data
        element_info = next(
            (
                e
                for e in elements_data
                if e["bbox"][0] <= norm_x <= e["bbox"][2]
                and e["bbox"][1] <= norm_y <= e["bbox"][3]
            ),
            None,
        )

        # If no direct match is found, find the closest element
        if element_info is None and elements_data:
            min_distance = float("inf")
            closest_element = None

            for element in elements_data:
                bbox = element["bbox"]
                # Calculate the center point of the bounding box
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Calculate the Euclidean distance from the click position to the center of the bounding box
                distance = ((norm_x - center_x) ** 2 + (norm_y - center_y) ** 2) ** 0.5

                # Update the closest element
                if distance < min_distance:
                    min_distance = distance
                    closest_element = element

            element_info = closest_element
            print(
                f"No direct match found. Using closest element with distance {min_distance:.4f}"
            )

        return element_info

    except Exception as e:
        print(f"Error in pos2id: {str(e)}")
        return None


def json2db(json_path: str):
    """
    Chain stored logs into the database
    :param json_path:
    :return:
    """
    # Initialize graph database connection
    db = Neo4jDatabase(
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
    )

    # Initialize vector database connection
    vector_store = VectorStore(
        api_key=config.PINECONE_API_KEY,
        dimension=2048,
        batch_size=2,
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generate task ID
    task_id = generate_short_md5(data["tsk"], 12)

    # Store page and element information
    pages_info = []  # Store all page information
    elements_info = []  # Store all element information

    # First loop: create all nodes and page->element relationships
    for step in data["history_steps"]:
        # Process page node
        page_properties = {
            "page_id": str(uuid4()),
            "description": "",
            "raw_page_url": step["source_page"],
            "timestamp": step["timestamp"],
            "other_info": json.dumps(
                {
                    "step": step["step"],
                    **(
                        {"task_info": {"task_id": task_id, "description": data["tsk"]}}
                        if step["step"] == 0
                        else {}
                    ),
                }
            ),
        }

        # Read element JSON file
        elements_path = Path(step["source_json"].replace("\\", "/"))
        with open(elements_path, "r", encoding="utf-8") as f:
            elements_data = json.load(f)
            page_properties["elements"] = json.dumps(elements_data)

        # Create page node
        db.create_page(page_properties)
        pages_info.append({"page_id": page_properties["page_id"], "step": step["step"]})

        # Modify element node processing logic
        tool_result = step["tool_result"]
        action_type = tool_result.get("action")

        # Get element information
        element_info = None
        clicked_element = tool_result.get("clicked_element")

        if action_type == "tap":
            element_info = pos2id(
                clicked_element["x"],
                clicked_element["y"],
                step["source_json"].replace("\\", "/"),
            )
        else:
            element_info = {"ID": "", "bbox": [], "type": "", "content": ""}

        if element_info:
            parameters = {
                k: v
                for k, v in tool_result.items()
                if k not in ["action", "device", "status"]
            }

            element_properties = {
                "element_id": str(uuid4()),
                "element_original_id": element_info.get("ID", ""),
                "description": "",
                "action_type": action_type,
                "parameters": json.dumps(parameters),
                "bounding_box": element_info.get("bbox", []),
                "other_info": json.dumps(
                    {
                        "type": element_info.get("type", ""),
                        "content": element_info.get("content", ""),
                    }
                ),
            }

            # Create element node
            db.create_element(element_properties)
            elements_info.append(
                {
                    "element_id": element_properties["element_id"],
                    "step": step["step"],
                    "action": step["recommended_action"],
                    "status": tool_result["status"],
                    "timestamp": step["timestamp"],
                }
            )

            # Establish element to page ownership relationship
            db.add_element_to_page(
                page_properties["page_id"], element_properties["element_id"]
            )

            # Process the visual features of the element and store them in the vector database
            if element_info.get("ID"):  # Only process valid element ID
                success = element2vector(
                    str(element_info["ID"]),
                    element_properties["element_id"],
                    json.dumps(elements_data),
                    step["source_page"],
                    vector_store,
                )
                if not success:
                    print(
                        f"Warning: Vector storage failed for element {element_info['ID']}"
                    )

    # Create final page node (if exists)
    if data.get("final_page"):
        final_page_properties = {
            "page_id": str(uuid4()),
            "description": "",
            "raw_page_url": data["final_page"].get("screenshot", ""),
            "timestamp": data["final_page"].get("timestamp", ""),
        }

        # Read final page element JSON
        if data["final_page"].get("page_json"):
            elements_path = Path(data["final_page"]["page_json"].replace("\\", "/"))
            with open(elements_path, "r", encoding="utf-8") as f:
                elements_data = json.load(f)
                final_page_properties["elements"] = json.dumps(elements_data)
        else:
            final_page_properties["elements"] = json.dumps(
                []
            )  # If no element data, set to empty list

        # Create final page node
        db.create_page(final_page_properties)
        pages_info.append(
            {"page_id": final_page_properties["page_id"], "step": "final"}
        )

    # Second loop: establish element->page leads_to relationship
    for i in range(len(elements_info)):
        current_element = elements_info[i]
        next_page = None

        # If it's the last element, point to the final page (if exists)
        if i == len(elements_info) - 1 and len(pages_info) > len(elements_info):
            next_page = pages_info[-1]  # Last page (final page)
        else:
            # Otherwise point to the next regular page
            next_page = next(
                (p for p in pages_info if p["step"] == current_element["step"] + 1),
                None,
            )

        if next_page:
            db.add_element_leads_to(
                current_element["element_id"],
                next_page["page_id"],
                action_name=current_element["action"],
                action_params={
                    "execution_result": current_element["status"],
                    "timestamp": current_element["timestamp"],
                },
            )

    return task_id


def element2vector(
    ID: str,
    element_id: str,
    elements_json: str,
    page_path: str,
    vector_store: VectorStore,
) -> bool:
    """
    Process the visual features of the element and store them in the vector database

    Parameters:
        ID: str, the ID of the element in the JSON
        element_id: str, the unique ID of the element in the graph database
        elements_json: str, the element JSON string
        page_path: str, the page image path
        vector_store: VectorStore, the vector database instance

    Returns:
        bool: Whether the storage was successful
    """
    try:
        # 1. Extract element image
        element_image = element_img(page_path, elements_json, int(ID))

        # 2. Extract visual features
        features = extract_features(element_image, "resnet50")

        # 3. Parse JSON string to get element information
        elements = json.loads(elements_json)
        target_element = next((e for e in elements if e.get("ID") == int(ID)), None)

        if target_element is None:
            raise ValueError(f"Element with ID {ID} not found")

        # 4. Prepare vector data
        vector_data = VectorData(
            id=element_id,
            values=features["features"][0],
            metadata={
                "original_id": str(ID),
                "bbox": target_element["bbox"],
                "type": target_element.get("type", ""),
                "content": target_element.get("content", ""),
            },
            node_type=NodeType.ELEMENT,
        )

        # 5. Store in vector database
        return vector_store.upsert_batch([vector_data])

    except Exception as e:
        print(f"Error processing element vector: {str(e)}")
        return False
