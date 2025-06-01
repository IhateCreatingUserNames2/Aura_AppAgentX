from .data.State import State
from .tool.screen_content import *
import json
import os
from typing import Tuple
import datetime



def capture_and_parse_page(state: State) -> State:
    """
    capture_and_parse_page

    Function Overview:
        1. Take a screenshot of the current device interface based on the device information in State and save it to the specified path;
        2. Call the page parsing tool to analyze the screenshot, obtaining the labeled image and corresponding JSON file;
        3. Write the screenshot path and parsed JSON file path into State and store them in relevant fields;
        4. Return the updated State.

    Parameters:
        - state (State): Context state machine, containing device information, historical operations, error records, etc.

    Returns:
        - State: Updated state dictionary, which includes the latest screenshot path, parsing result path, etc.
    """

    # 1. Take screenshot and save
    device_id = state.get("device", "emulator")
    app_name = state.get("app_name", "unknown_app")

    screenshot_result = take_screenshot.invoke(
        {
            "device": device_id,
            "save_dir": "./log/screenshots",
            "app_name": app_name,
            "step": state["step"] + 1,
        }
    )

    saved_path = None

    if "failed" in screenshot_result:
        # Log error and print log when screenshot fails
        error_msg = screenshot_result
        print(f"[capture_and_parse_page] Screenshot failed: {error_msg}")
        state["errors"].append(
            {"step": state["step"], "tool": "take_screenshot", "error_msg": error_msg}
        )
    else:
        saved_path = screenshot_result
        print(f"[capture_and_parse_page] Screenshot path: {saved_path}")

    # 2. Call the page parsing tool
    if saved_path:
        parse_result = screen_element.invoke({"image_path": saved_path})
        if "error" not in parse_result:
            labeled_img = parse_result["labeled_image_path"]
            parsed_json = parse_result["parsed_content_json_path"]
            state["current_page_screenshot"] = saved_path
            state["current_page_json"] = parsed_json

            state["page_history"].append(labeled_img)

            state["tool_results"].append(
                {"tool_name": "screen_element", "result": parse_result}
            )
        else:
            error_info = parse_result["error"]
            print(f"[capture_and_parse_page] Screen parse failed: {error_info}")
            state["errors"].append(
                {
                    "step": state["step"],
                    "tool": "screen_element",
                    "error_msg": error_info,
                }
            )

    return state


def element_number_to_coords(state: State, element_id: int) -> Tuple[int, int]:
    """
    Given the element_id (ID of the element), find the corresponding element's 'bbox' in the JSON file pointed to by state['current_page_json']
    (format: [x1, y1, x2, y2], usually relative coordinates), and calculate the pixel-level coordinates (center_x, center_y) of the element on the screen,
    returning a tuple (x, y).

    Parameters:
        - state (State): Context state machine, must contain the following information:
            1. current_page_json: Path to the current page's JSON file (contains element information)
            2. device_info: Device information dictionary, must include 'width', 'height' indicating screen pixel resolution
        - element_id (int): The ID of the element to be found

    Returns:
        - (x, y): Returns the center pixel coordinates of the element on the screen in integer format

    Exception or error handling:
        - If the JSON file does not exist or cannot be parsed, an exception is raised or written to state['errors']
        - If the corresponding element_id is not found, an exception is raised or written to state['errors']
        - If device_info is missing width or height, an exception is raised or written to state['errors']
    """

    # 1. Validate if the current page JSON file exists
    json_file_path = state.get("current_page_json")
    if not json_file_path or not os.path.isfile(json_file_path):
        error_msg = (
            f"[element_number_to_coords] JSON file does not exist: {json_file_path}"
        )
        print(error_msg)
        state["errors"].append(
            {
                "step": state["step"],
                "func": "element_number_to_coords",
                "error_msg": error_msg,
            }
        )
        raise FileNotFoundError(error_msg)

    # 2. Validate device information and get screen resolution
    device_info = state.get("device_info", {})
    screen_width = device_info.get("width")
    screen_height = device_info.get("height")

    if not screen_width or not screen_height:
        error_msg = "[element_number_to_coords] Missing device screen resolution information (device_info['width'] or device_info['height'])."
        print(error_msg)
        state["errors"].append(
            {
                "step": state["step"],
                "func": "element_number_to_coords",
                "error_msg": error_msg,
            }
        )
        raise ValueError(error_msg)

    # 3. Read JSON file
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        error_msg = f"[element_number_to_coords] JSON file parsing failed: {str(e)}"
        print(error_msg)
        state["errors"].append(
            {
                "step": state["step"],
                "func": "element_number_to_coords",
                "error_msg": error_msg,
            }
        )
        raise e

    # 4. Find the element matching the element_id in the JSON data
    target_item = None
    for item in json_data:
        if item.get("ID") == element_id:
            target_item = item
            break

    if not target_item:
        error_msg = f"[element_number_to_coords] Element ID={element_id} not found."
        print(error_msg)
        state["errors"].append(
            {
                "step": state["step"],
                "func": "element_number_to_coords",
                "error_msg": error_msg,
            }
        )
        raise ValueError(error_msg)

    # 5. Read its bbox and calculate the screen center coordinates
    bbox = target_item.get("bbox")
    if not bbox or len(bbox) != 4:
        error_msg = f"[element_number_to_coords] Found element ID={element_id}, but bbox is invalid or missing: {bbox}"
        print(error_msg)
        state["errors"].append(
            {
                "step": state["step"],
                "func": "element_number_to_coords",
                "error_msg": error_msg,
            }
        )
        raise ValueError(error_msg)

    x1, y1, x2, y2 = bbox
    # Get center point
    center_x_rel = (x1 + x2) / 2.0
    center_y_rel = (y1 + y2) / 2.0

    # Convert to absolute pixel coordinates
    pixel_x = int(center_x_rel * screen_width)
    pixel_y = int(center_y_rel * screen_height)
    print(
        f"[element_number_to_coords] Element {element_id} center: ({pixel_x}, {pixel_y})"
    )
    return pixel_x, pixel_y


def single_human_explor(state: State, action: str, **kwargs) -> State:
    """
    single_human_explor function

    Function Overview:
        1. Based on the passed action and related parameters (kwargs), call the screen_action function to perform the corresponding screen action;
        2. Write the action execution result into State for subsequent processes or debugging;
        3. Take a screenshot of the current device interface based on the device information in State and save it to the specified path;
        4. Call the page parsing tool to analyze the screenshot, obtaining the labeled image and corresponding JSON file;
        5. Write the screenshot path and parsed JSON file path into State for subsequent processes.

    Parameters:
        - state (State): Context state machine, containing device information, historical operations, error records, etc.
        - action (str): The type of screen action to be executed this time, such as click, text, swipe, etc.
        - kwargs (dict): Other optional parameters, for example:
            - element_number (int): When clicking or long-pressing an element, this index may be needed to determine coordinates
            - text_input (str): The text to be input when performing a text action
            - swipe_direction (str): When performing a swipe action, specify the direction ("up", "down", "left", "right")
            - start (tuple): Starting coordinates for swipe_precise operation (x_start, y_start)
            - end (tuple): Ending coordinates for swipe_precise operation (x_end, y_end)
            - duration (int): Duration of long press or precise swipe (milliseconds), default 1000 or 400
            - x (int), y (int): Screen coordinates for click, long press, etc.

    Returns:
        - State: Updated state dictionary, which includes the latest screenshot path, parsing result path, current action execution result, etc.
    """

    print(f"[human_explor] action: {action}, kwargs: {kwargs}")

    # === 1. Execute action operation (screen_action) ===
    screen_action_result = None
    if action in ["tap", "text", "long_press", "swipe", "swipe_precise", "back", "tap"]:

        x, y = element_number_to_coords(state, kwargs.get("element_number"))
        text_input = kwargs.get("text_input")
        swipe_direction = kwargs.get("swipe_direction")
        dist = kwargs.get("dist", "medium")
        quick = kwargs.get("quick", False)
        duration = kwargs.get("duration", 1000)
        start_coords = kwargs.get("start")
        end_coords = kwargs.get("end")

        params_dict = {"device": state.get("device", "emulator"), "action": action}

        if action == "back":
            screen_action_result = screen_action.invoke(params_dict)

        elif action == "tap":
            if x is not None and y is not None:
                params_dict.update({"x": x, "y": y})
                screen_action_result = screen_action.invoke(params_dict)
            else:
                msg = "[single_human_explor] Missing x,y for click action."
                print(msg)
                state["errors"].append(
                    {"step": state["step"], "tool": "screen_action", "error_msg": msg}
                )

        elif action == "text":
            if text_input:
                params_dict.update({"input_str": text_input})
                screen_action_result = screen_action.invoke(params_dict)
            else:
                msg = "[single_human_explor] Missing text_input for text action."
                print(msg)
                state["errors"].append(
                    {"step": state["step"], "tool": "screen_action", "error_msg": msg}
                )

        elif action == "long_press":
            if x is not None and y is not None:
                params_dict.update({"x": x, "y": y, "duration": duration})
                screen_action_result = screen_action.invoke(params_dict)
            else:
                msg = "[single_human_explor] Missing x,y for long_press action."
                print(msg)
                state["errors"].append(
                    {"step": state["step"], "tool": "screen_action", "error_msg": msg}
                )

        elif action == "swipe":
            if x is not None and y is not None and swipe_direction:
                params_dict.update(
                    {
                        "x": x,
                        "y": y,
                        "direction": swipe_direction,
                        "dist": dist,
                        "quick": quick,
                    }
                )
                screen_action_result = screen_action.invoke(params_dict)
            else:
                msg = "[single_human_explor] Missing x,y or swipe_direction for swipe action."
                print(msg)
                state["errors"].append(
                    {"step": state["step"], "tool": "screen_action", "error_msg": msg}
                )

        elif action == "swipe_precise":
            if start_coords and end_coords:
                params_dict.update(
                    {"start": start_coords, "end": end_coords, "duration": duration}
                )
                screen_action_result = screen_action.invoke(params_dict)
            else:
                msg = (
                    "[single_human_explor] Missing start,end for swipe_precise action."
                )
                print(msg)
                state["errors"].append(
                    {"step": state["step"], "tool": "screen_action", "error_msg": msg}
                )

        if screen_action_result:
            try:
                action_dict = json.loads(screen_action_result)
            except json.JSONDecodeError:
                action_dict = {"raw_result": screen_action_result}

            state["tool_results"].append(
                {"tool_name": "screen_action", "action_result": action_dict}
            )
            step_record = {
                "step": state["step"],
                "recommended_action": f"Executing {action} operation with parameters {kwargs}",
                "tool_result": {
                    "action": action,
                    "device": state.get("device", "emulator"),
                    "clicked_element": (
                        {"x": x, "y": y} if action in ["tap", "long_press"] else None
                    ),
                    "status": "success" if screen_action_result else "failed",
                    **action_dict,
                },
                "source_page": state.get("current_page_screenshot"),
                "source_json": state.get("current_page_json"),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            state["history_steps"].append(step_record)

    else:
        print(f"[human_explor] Action '{action}' is not recognized or not handled.")
        state["errors"].append(
            {
                "step": state["step"],
                "action": action,
                "error_msg": f"Unsupported action: {action}",
            }
        )
    state = capture_and_parse_page(state)
    state["step"] += 1
    return state
