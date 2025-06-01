import base64
import datetime
import json
import os
import subprocess
from time import sleep
from typing import Dict
import requests
from langchain_core.tools import tool
from .. import config


# Define a function to execute ADB commands
def execute_adb(adb_command):
    result = subprocess.run(
        adb_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"Command execution failed: {adb_command}")
    print(result.stderr)
    return "ERROR"


def list_all_devices() -> list:
    """
    List all devices currently connected via ADB (Android Debug Bridge).

    Returns:
        - Success: Returns a list of device IDs, each representing a connected device.
        - Failure: Returns an empty list.

    Return examples:
        1. When devices are connected:
            ["emulator-5551", "device123456"]
        2. When no devices are connected:
            []
    """
    adb_command = "adb devices"
    device_list = []
    result = execute_adb(adb_command)
    if result != "ERROR":
        devices = result.split("\n")[1:]
        for d in devices:
            device_list.append(d.split()[0])
    return device_list


@tool
def get_device_size(device: str = "emulator") -> dict | str:
    """
    Get the screen size (width and height) of a mobile device (Android emulator or real device).
    Parameters:
        - device (str): Specify the target device ID, default is "emulator".

    Returns:
        - Success: Returns the result of the operation, which is the screen width and height (in pixels).
        - Returns error information.

    """
    adb_command = f"adb -s {device} shell wm size"
    result = execute_adb(adb_command)
    if result != "ERROR":
        size_str = result.split(": ")[1]
        width, height = map(int, size_str.split("x"))
        return {"width": width, "height": height}
    return "Failed to get device size. Please check device connection or permissions."


# Define a screenshot tool
@tool
def take_screenshot(
    device: str = "emulator",
    save_dir: str = "./log/screenshots",
    app_name: str = None,
    step: int = 0,
) -> str:
    """
    Take a screenshot of the specified mobile device (Android emulator or real device) and save it to a directory organized by application.

    Parameters:
        - device (str): Specify the target device ID, default is "emulator". You can view connected devices using the `list_all_devices` tool.
        - save_dir (str): Directory path to save the screenshot locally, default is "./screenshots" in the current directory.
        - app_name (str): Name of the current application, used to organize subdirectories for saving screenshots.
        - step (int): Step number of the current operation, used to generate the filename.

    Returns:
        - Success: Returns the specific path string where the screenshot is saved.
        - Failure: Returns an error message string, such as "Screenshot failed, please check device connection or permissions".
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if app_name is None:
        app_name = "unknown_app"

    # Create a subdirectory organized by application
    app_dir = os.path.join(save_dir, app_name)
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate a filename including the application name, step number, and timestamp
    if step is not None:
        filename = f"{app_name}_step{step}_{timestamp}.png"
    else:
        filename = f"{app_name}_{timestamp}.png"

    screenshot_file = os.path.join(app_dir, filename)
    remote_file = f"/sdcard/{filename}"

    # Construct ADB commands
    cap_command = f"adb -s {device} shell screencap -p {remote_file}"
    pull_command = f"adb -s {device} pull {remote_file} {screenshot_file}"
    delete_command = f"adb -s {device} shell rm {remote_file}"

    sleep(3)
    # Execute screenshot command
    try:
        if execute_adb(cap_command) != "ERROR":
            if execute_adb(pull_command) != "ERROR":
                execute_adb(
                    delete_command
                )  # Delete temporary screenshot file from device
                return f"{screenshot_file}"
    except Exception as e:
        return f"Screenshot failed, error information: {str(e)}"

    return "Screenshot failed. Please check device connection or permissions."


@tool
def screen_element(image_path: str) -> Dict:
    """
    Call the page understanding tool interface, upload the screenshot file and receive the parsing result, save the labeled image and parsing content locally, and return the file path.
    Parameters:
        - image_path (str): File path of the screenshot (local path).

    Returns:
        - Success: Returns a dictionary containing the labeled image save path and parsed content JSON file path.
        - Failure: Returns a dictionary containing error information.
    """
    api_url = f"{config.Omni_URI}/process_image/"
    # Check if the screenshot file exists
    if not os.path.exists(image_path):
        return {"error": "Screenshot file does not exist. Please check the path."}

    # If save_dir is not provided, dynamically generate based on image_path
    save_dir = os.path.join(os.path.dirname(image_path), "processed_images")

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Read the screenshot file
    try:
        with open(image_path, "rb") as file:
            files = [("file", (os.path.basename(image_path), file, "image/png"))]
            response = requests.post(api_url, files=files)

        # Check response status
        if response.status_code != 200:
            return {
                "error": f"Interface call failed, status code: {response.status_code}, information: {response.text}"
            }

        # Parse response data
        data = response.json()
        if data.get("status") != "success":
            return {
                "error": "Interface returned failed status. Please check interface logic.",
                "details": data,
            }

        # Get parsed content and labeled image data
        parsed_content = data.get("parsed_content", [])
        labeled_image_base64 = data.get("labeled_image", "")
        elapsed_time = data.get("e_time", None)

        # Save the labeled image
        if labeled_image_base64:
            labeled_image_data = base64.b64decode(labeled_image_base64)
            labeled_image_path = os.path.join(
                save_dir, f"labeled_{os.path.basename(image_path)}"
            )
            with open(labeled_image_path, "wb") as labeled_image_file:
                labeled_image_file.write(labeled_image_data)
        else:
            return {
                "error": "Labeled image data missing, unable to save labeled image."
            }

        # Save parsed content to JSON file
        json_file_path = os.path.join(
            save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json"
        )
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json_file.write(json.dumps(parsed_content, ensure_ascii=False, indent=4))

        # Return parsed result file path
        return {
            "labeled_image_path": labeled_image_path,
            "parsed_content_json_path": json_file_path,
            "elapsed_time": elapsed_time,
        }

    except Exception as e:
        return {"error": f"An exception occurred during tool execution: {str(e)}"}


# Define action tools, including tap, back, type, swipe, long press, and drag
@tool
def screen_action(
    device: str = "emulator",
    action: str = "tap",
    x: int = None,
    y: int = None,
    input_str: str = None,
    duration: int = 1000,
    direction: str = None,
    dist: str = "medium",
    quick: bool = False,
    start: tuple = None,
    end: tuple = None,
) -> str:
    """
    Tool name: screen_action

    Tool function:
        Perform screen operations on mobile devices (Android emulator or real device), including tap, back, type text, swipe, long press, and drag.

    Parameters:
        - device (str): Specify the target device ID, default is "emulator".
        - action (str): Specify the type of screen operation to perform. Supports the following operations:
            - "tap": Tap the specified coordinates on the screen.
                Requires parameters: x, y
            - "back": Back key operation.
                No additional parameters required.
            - "text": Enter text on the screen.
                Requires parameter: input_str
            - "long_press": Long press the specified coordinates on the screen.
                Requires parameters: x, y, duration (default 1000 milliseconds)
            - "swipe": Swipe operation, supports four directions ("up", "down", "left", "right").
                Requires parameters: x, y, direction, dist (default "medium"), quick (default False)
            - "swipe_precise": Precise swipe, swipe from the specified start point to the specified end point.
                Requires parameters: start, end, duration (default 400 milliseconds)

    Returns:
        Returns a JSON string including the following fields:
        - "status": "success" or "error"
        - "action": Type of operation performed
        - "device": Device ID
        - Other fields appended according to the operation type (e.g., clicked coordinates, input text, swipe start and end points, etc.).
    """
    try:
        adb_command = None
        result_data = {"action": action, "device": device}

        if action == "back":
            adb_command = f"adb -s {device} shell input keyevent KEYCODE_BACK"

        elif action == "tap":
            if x is None or y is None:
                return json.dumps(
                    {
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for click action (x, y)",
                    }
                )
            adb_command = f"adb -s {device} shell input tap {x} {y}"
            # Record information of the clicked element (assuming x, y represent the position of the clicked element)
            result_data["clicked_element"] = {"x": x, "y": y}

        elif action == "text":
            if not input_str:
                return json.dumps(
                    {
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameter for text action (input_str)",
                    }
                )
            sanitized_input_str = input_str.replace(" ", "%s").replace("'", "")
            adb_command = f"adb -s {device} shell input text {sanitized_input_str}"
            result_data["input_str"] = input_str

        elif action == "long_press":
            if x is None or y is None:
                return json.dumps(
                    {
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for long_press action (x, y)",
                    }
                )
            adb_command = (
                f"adb -s {device} shell input swipe {x} {y} {x} {y} {duration}"
            )
            result_data["long_press"] = {"x": x, "y": y, "duration": duration}

        elif action == "swipe":
            if x is None or y is None or direction is None:
                return json.dumps(
                    {
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for swipe action (x, y, direction)",
                    }
                )
            unit_dist = 100  # Swipe base distance
            offset_x, offset_y = 0, 0
            if direction == "up":
                offset_y = -2 * unit_dist if dist == "medium" else -3 * unit_dist
            elif direction == "down":
                offset_y = 2 * unit_dist if dist == "medium" else 3 * unit_dist
            elif direction == "left":
                offset_x = -2 * unit_dist if dist == "medium" else -3 * unit_dist
            elif direction == "right":
                offset_x = 2 * unit_dist if dist == "medium" else 3 * unit_dist
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Invalid direction for swipe",
                    }
                )
            swipe_duration = 100 if quick else 400
            adb_command = f"adb -s {device} shell input swipe {x} {y} {x + offset_x} {y + offset_y} {swipe_duration}"
            result_data["swipe"] = {
                "start": (x, y),
                "end": (x + offset_x, y + offset_y),
                "duration": swipe_duration,
                "direction": direction,
            }

        elif action == "swipe_precise":
            if not start or not end:
                return json.dumps(
                    {
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for swipe_precise action (start, end)",
                    }
                )
            start_x, start_y = start
            end_x, end_y = end
            adb_command = f"adb -s {device} shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}"
            result_data["swipe_precise"] = {
                "start": start,
                "end": end,
                "duration": duration,
            }

        else:
            return json.dumps(
                {
                    "status": "error",
                    "action": action,
                    "device": device,
                    "message": "Invalid action",
                }
            )

        # Execute ADB command
        ret = execute_adb(adb_command)
        if ret is not None and "ERROR" not in ret.upper():
            result_data["status"] = "success"
        else:
            result_data["status"] = "error"
            result_data["message"] = f"ADB command execution failed: {ret}"

        return json.dumps(result_data, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"status": "error", "action": action, "device": device, "message": str(e)},
            ensure_ascii=False,
        )
