import datetime
import json
import os
import sys
import time
import traceback
from queue import Queue, Empty

APPAGENTX_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root (Aura2)
PROJECT_ROOT = os.path.dirname(APPAGENTX_DIR)

# Add the project root to sys.path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import threading
from queue import Queue
import gradio as gr
from . import config
from .explor_auto import run_task
from .chain_evolve import evolve_chain_to_action
from .chain_understand import process_and_update_chain, Neo4jDatabase
from .data.State import State
from .data.data_storage import state2json, json2db
from .explor_human import single_human_explor, capture_and_parse_page
from .tool.screen_content import list_all_devices, get_device_size

auto_log_storage = []  # Global log storage for automatic exploration
auto_page_storage = []  # Global page history storage for automatic exploration
user_log_storage = []  # Global log storage for user exploration
user_page_storage = []  # Global page history storage for user exploration
temp_state = None  # Temporary state storage for saving user exploration state


def update_inputs(action):
    if action == "tap" or action == "long press":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif action == "text":
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif action == "swipe":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def get_adb_devices():
    devices = list_all_devices()
    return devices if devices else ["No devices found"]


def initialize_device(device, task_info):
    global temp_state
    if not task_info:
        return "Error: Task Information cannot be empty."
    if not device or device == "No devices found":
        return "Error: Please select a valid ADB device."
    device_size = get_device_size.invoke({"device": device})
    temp_state = State(
        tsk=task_info,
        app_name="",
        completed=False,
        step=0,
        history_steps=[],
        page_history=[],
        current_page_screenshot=None,
        recommend_action="",
        clicked_elements=[],
        action_reflection=[],
        tool_results=[],
        device=device,
        device_info=device_size,
        context=[],
        errors=[],
        current_page_json=None,
        callback=None,
    )
    return f"Device: {device}, Task Info: {task_info} initialized."


def auto_exploration():
    global auto_log_storage, auto_page_storage, temp_state

    if not temp_state:
        auto_log_storage.append("Error: Please initialize device and task info first.")
        yield "\n".join(auto_log_storage), auto_page_storage
        return

    q = Queue()
    final_state_queue = Queue()

    def callback(state, node_name=None, info=None):
        auto_log_storage.append("--" * 10)
        auto_log_storage.append(
            f"Current execution step: {state['step']}, Completed node: {node_name}"
        )
        if info and isinstance(info, dict):
            auto_log_storage.append("Additional information:")
            for k, v in info.items():
                auto_log_storage.append(f"   {k}: {v}")

        if state.get("tool_results"):
            for tool_result in state["tool_results"]:
                if isinstance(tool_result, dict):
                    result = tool_result.get("result", {})
                    if isinstance(result, dict):
                        labeled_image = result.get("labeled_image_path")
                        if labeled_image and labeled_image not in auto_page_storage:
                            auto_page_storage.append(labeled_image)

        q.put(("\n".join(auto_log_storage), auto_page_storage))

    def run_exploration():
        final_state = run_task(temp_state, callback)
        final_state_queue.put(final_state)

    # Start the exploration task in a new thread
    t = threading.Thread(target=run_exploration)
    t.start()

    # Continuously get status updates from the queue
    while t.is_alive() or not q.empty():
        try:
            msg, pages = q.get(timeout=1)
            yield msg, pages
        except:
            pass

    auto_log_storage.append("Exploration finished.")

    # Get the final state from the queue
    try:
        final_state = final_state_queue.get(timeout=5)  # Wait for the final state
        # Convert the result to JSON format and store it
        state2json_result = state2json(final_state)
        auto_log_storage.append(state2json_result)
    except:
        auto_log_storage.append("Error: Failed to get final state")

    yield "\n".join(auto_log_storage), auto_page_storage


def user_exploration(action, element_number, text_input, swipe_direction):
    global user_log_storage, user_page_storage, temp_state

    if not temp_state:
        user_log_storage.append("Error: Please initialize device and task info first.")
        return "\n".join(user_log_storage), user_page_storage

    # Call the single human exploration logic
    temp_state = single_human_explor(
        temp_state,
        action,
        element_number=element_number,
        text_input=text_input,
        swipe_direction=swipe_direction,
    )

    # Log additional operation details
    log_entry = f"Step: {temp_state.get('step', 0)} | Action: {action}"
    if action in ["tap", "long press"]:
        log_entry += f" on element {element_number}"
    elif action == "text":
        log_entry += f" with text input '{text_input}' on element {element_number}"
    elif action == "swipe":
        log_entry += f" swiped {swipe_direction} on element {element_number}"

    # Check task completion status
    if temp_state.get("completed", False):
        log_entry += " | Task completed."
    else:
        log_entry += " | Task in progress."

    user_log_storage.append(log_entry)

    # Check for updates in page_history and update page storage
    if "page_history" in temp_state:
        for page in temp_state["page_history"]:
            if page not in user_page_storage:
                user_page_storage.append(page)

    # Add the latest labeled screenshot to the page storage
    if temp_state.get("history_steps"):
        latest_step = temp_state["history_steps"][-1]
        if latest_step.get("source_json"):
            for tool_result in reversed(temp_state["tool_results"]):
                if tool_result.get("tool_name") == "screen_element":
                    labeled_image_path = tool_result["result"].get("labeled_image_path")
                    if (
                        labeled_image_path
                        and labeled_image_path not in user_page_storage
                    ):
                        user_page_storage.append(labeled_image_path)
                    break

    return "\n".join(user_log_storage), user_page_storage


def self_evolution(task, data):
    return f"Task: {task} with Data: {data} is evolving."


def get_json_files(directory: str = "./log/json_state") -> list:
    """Get all JSON files and their details in the specified directory"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        json_files = []
        for file in os.listdir(directory):
            if file.endswith(".json"):
                file_path = os.path.join(directory, file)
                mod_time = os.path.getmtime(file_path)
                mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        steps_count = len(content.get("history_steps", []))
                        task_desc = content.get("tsk", "N/A")
                        app_name = content.get("app_name", "N/A")
                except Exception as e:
                    print(f"Error reading JSON file {file}: {str(e)}")
                    steps_count = 0
                    task_desc = "Error reading file"
                    app_name = "N/A"

                json_files.append(
                    {
                        "name": file,
                        "path": file_path,
                        "modified": mod_time_str,
                        "steps": steps_count,
                        "task": task_desc,
                        "app": app_name,
                    }
                )
        return sorted(json_files, key=lambda x: x["modified"], reverse=True)
    except Exception as e:
        print(f"Error reading directory: {str(e)}")
        return []


with gr.Blocks(
    css="#json_files_table table { border-collapse: collapse; width: 100%; } #json_files_table thead { background-color: #f3f4f6; } #json_files_table th { padding: 10px; text-align: left; border-bottom: 2px solid #e5e7eb; } #json_files_table td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; } #json_files_table tr:hover { background-color: #f9fafb; } #json_files_table .scroll-hide { overflow-x: hidden !important; } #json_files_table { user-select: none; }"
) as demo:
    with gr.Tabs():
        # Tab 1: Initialization
        with gr.Tab("Initialization"):
            gr.Markdown("### Initialization Page")
            devices_output = gr.Textbox(
                label="Available ADB Devices", interactive=False
            )
            refresh_button = gr.Button("Refresh Device List")
            device_selector = gr.Radio(
                label="Select ADB Device", choices=[]
            )  # Updated dynamically
            task_info_input = gr.Textbox(
                label="Set Task Information", placeholder="e.g., Task Description"
            )
            init_button = gr.Button("Initialize")
            init_output = gr.Textbox(label="Initialization Output", interactive=False)

            def update_devices():
                devices = get_adb_devices()
                return gr.update(choices=devices), "\n".join(devices)

            refresh_button.click(
                update_devices, inputs=[], outputs=[device_selector, devices_output]
            )

            init_button.click(
                initialize_device,
                inputs=[device_selector, task_info_input],
                outputs=[init_output],
            )

        # Tab 2: Auto Exploration
        with gr.Tab("Auto Exploration"):
            gr.Markdown("### Auto Exploration Page")
            with gr.Row():
                with gr.Column():
                    exploration_output = gr.TextArea(
                        label="Exploration Log", interactive=False
                    )
                    explore_button = gr.Button("Start Exploration")
                    clear_images_button = gr.Button(
                        "Clear Images", elem_id="clear-images-button"
                    )

                with gr.Column():
                    screenshot_gallery_auto = gr.Gallery(
                        label="Screenshots", height=700
                    )
            explore_button.click(
                fn=auto_exploration,
                outputs=[
                    exploration_output,
                    screenshot_gallery_auto,
                ],
                queue=True,
                api_name="explore",
            )

            def clear_images():
                return []

            clear_images_button.click(
                fn=clear_images,
                inputs=[],
                outputs=[screenshot_gallery_auto],
            )

        # Tab 3: User Exploration
        with gr.Tab("User Exploration"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### User Exploration Page")
                    with gr.Row():
                        start_button = gr.Button(
                            "Start Session", elem_id="start-button"
                        )
                        stop_button = gr.Button(
                            "Stop Session", elem_id="stop-button", interactive=False
                        )
                    action = gr.Radio(
                        ["tap", "text", "long press", "swipe", "wait"],
                        label="Action",
                        interactive=False,
                    )
                    element_number = gr.Number(
                        label="Element Number",
                        precision=0,
                        visible=False,
                        interactive=False,
                    )
                    text_input = gr.Textbox(
                        label="Text Input", visible=False, interactive=False
                    )
                    swipe_direction = gr.Radio(
                        ["up", "down", "left", "right"],
                        label="Swipe Direction",
                        visible=False,
                        interactive=False,
                    )
                    action_button = gr.Button("Perform Action", interactive=False)
                    human_demo_output = gr.Textbox(
                        label="Action Output", interactive=False
                    )

                with gr.Column():
                    screenshot_gallery_user = gr.Gallery(
                        label="Screenshots", height=700
                    )

            action.change(
                update_inputs,
                inputs=[action],
                outputs=[element_number, text_input, swipe_direction],
            )

            def start_session():
                global temp_state
                if not temp_state:
                    return None

                temp_state = capture_and_parse_page(
                    temp_state
                )  # Capture and initialize the first screenshot
                # Check for updates in page_history and update page storage
                if "page_history" in temp_state:
                    for page in temp_state["page_history"]:
                        if page not in user_page_storage:
                            user_page_storage.append(page)
                return (
                    gr.update(interactive=False),  # Disable start button
                    gr.update(interactive=True),  # Enable stop button
                    gr.update(interactive=True),  # Enable action selection
                    gr.update(interactive=True),  # Enable action button
                    gr.update(interactive=True),  # Enable Element Number
                    gr.update(interactive=True),  # Enable Text Input
                    gr.update(interactive=True),  # Enable Swipe Direction
                    gr.update(interactive=True),  # Enable store_to_db_btn
                    human_demo_output,  # Return output box
                    user_page_storage,  # Return page history
                )

            def stop_session():
                global temp_state
                if temp_state:
                    temp_state["completed"] = True
                user_log_storage.append("Session stopped.")
                user_page_storage.clear()
                state2json_result = state2json(temp_state)
                user_log_storage.append(state2json_result)
                return (
                    gr.update(interactive=True),  # Enable start button
                    gr.update(interactive=False),  # Disable stop button
                    gr.update(interactive=False),  # Disable action selection
                    gr.update(interactive=False),  # Disable action button
                    gr.update(interactive=False),  # Disable Element Number
                    gr.update(interactive=False),  # Disable Text Input
                    gr.update(interactive=False),  # Disable Swipe Direction
                    gr.update(interactive=False),  # Disable store_to_db_btn
                    "\n".join(user_log_storage),  # Return logs
                    user_page_storage,  # Return page history
                )

            action_button.click(
                user_exploration,
                inputs=[action, element_number, text_input, swipe_direction],
                outputs=[human_demo_output, screenshot_gallery_user],
            )

            start_button.click(
                start_session,
                inputs=[],
                outputs=[
                    start_button,
                    stop_button,
                    action,
                    action_button,
                    element_number,
                    text_input,
                    swipe_direction,
                    human_demo_output,
                    screenshot_gallery_user,
                ],
            )

            stop_button.click(
                stop_session,
                inputs=[],
                outputs=[
                    start_button,
                    stop_button,
                    action,
                    action_button,
                    element_number,
                    text_input,
                    swipe_direction,
                    human_demo_output,
                    screenshot_gallery_user,
                ],
            )

        # Tab 4: Chain Understanding & Evolution
        with gr.Tab("Chain Understanding & Evolution"):
            with gr.Column():
                with gr.Row():
                    # Left side: File selection
                    with gr.Column(scale=1):
                        json_files = gr.Radio(
                            label="Select Operation Record File",
                            choices=[],
                            value=None,
                            interactive=True,
                            elem_id="json_files_radio",
                        )

                    # Right side: File details
                    with gr.Column(scale=1):
                        file_info = gr.Markdown(
                            value="Please select a file to view details",
                            visible=True,
                            elem_id="file_info",
                        )

                evolution_log = gr.TextArea(
                    label="Execution Log", interactive=False, lines=8
                )

                with gr.Row():
                    refresh_btn = gr.Button(
                        "1. Refresh File List", variant="secondary", scale=1
                    )
                    store_to_db_btn = gr.Button(
                        "2. Store Record to Database", variant="primary", scale=1
                    )
                    understand_btn = gr.Button(
                        "3. Understand Operation Path", variant="primary", scale=1
                    )
                    generate_btn = gr.Button(
                        "4. Generate Advanced Path", variant="primary", scale=1
                    )

            # Store file list global variable
            current_files = []
            # Store current task ID
            current_task_id = None

            def format_file_choice(file_info):
                """Format file information as option text"""
                return file_info["name"]  # Return only file name

            def format_file_details(file_info):
                """Format file details"""
                return f"""
### File Details
- **File Name**: {file_info['name']}
- **Modified Time**: {file_info['modified']}
- **Step Count**: {file_info['steps']}
- **Task Description**: {file_info['task']}
- **Application Name**: {file_info['app']}
"""

            def update_file_list():
                """Update file list and return options"""
                global current_files
                try:
                    files = get_json_files()
                    current_files = files  # Update global variable

                    if not files:
                        return (
                            gr.update(choices=[], value=None),
                            "No JSON files found in directory.",
                            "No JSON files found in directory.",
                        )

                    # Generate option list
                    choices = [format_file_choice(f) for f in files]
                    # Default select first file
                    default_choice = choices[0] if choices else None

                    return (
                        gr.update(choices=choices, value=default_choice),
                        (
                            on_file_select(default_choice)
                            if default_choice
                            else "Please select a file to view details"
                        ),
                        (
                            "File list updated. First file selected as default."
                            if default_choice
                            else "File list updated, but no usable files."
                        ),
                    )
                except Exception as e:
                    import traceback

                    error_msg = (
                        f"Error updating file list: {str(e)}\n{traceback.format_exc()}"
                    )
                    return (
                        gr.update(choices=[], value=None),
                        error_msg,
                        error_msg,
                    )

            def on_file_select(choice):
                """When file is selected, update file details"""
                global current_files
                if not choice or not current_files:
                    return "Please select a file to view details"

                # Find corresponding file information based on selected text
                for file in current_files:
                    if format_file_choice(file) == choice:
                        return format_file_details(file)

                return "Failed to retrieve file information"

            # Bind refresh button event
            refresh_btn.click(
                fn=update_file_list, outputs=[json_files, file_info, evolution_log]
            )

            # Initialize with automatic load of file list
            demo.load(
                fn=update_file_list, outputs=[json_files, file_info, evolution_log]
            )

            # Define a function to refresh task ID from current file
            def refresh_task_id_from_file(selected_file):
                global current_files, current_task_id

                if not selected_file:
                    return "No file selected, cannot get task ID"

                # Get file path
                file_path = None
                for file in current_files:
                    if format_file_choice(file) == selected_file:
                        file_path = file["path"]
                        break

                if not file_path:
                    return "Failed to retrieve selected file path"

                try:
                    # Read JSON file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = json.load(f)

                    # Extract task ID from filename
                    import re

                    file_name = os.path.basename(file_path)
                    match = re.search(r"state_(\d+)_", file_name)
                    if match:
                        date_str = match.group(1)
                        # Generate a temporary task ID, format as "temp_" + date part
                        temp_task_id = f"temp_{date_str}"
                        current_task_id = temp_task_id
                        return f"Temporary task ID generated from filename: {temp_task_id}, Note: This is not the actual ID stored in the database"

                    return "Failed to extract date information to generate temporary task ID"
                except Exception as e:
                    return f"Error extracting task ID: {str(e)}"

            # Modify Radio component change event, update file details and attempt to get task ID
            def update_file_and_task_id(selected_file):
                file_details = on_file_select(selected_file)
                task_id_info = refresh_task_id_from_file(selected_file)
                return file_details, task_id_info

            json_files.change(
                fn=update_file_and_task_id,
                inputs=[json_files],
                outputs=[file_info, evolution_log],
            )

            # Define function to store to database
            def store_to_database(selected_file):
                global current_files, current_task_id

                if not selected_file:
                    return "Error: Please select a file first"

                # Find corresponding file information based on selected text
                file_path = None
                for file in current_files:
                    if format_file_choice(file) == selected_file:
                        file_path = file["path"]
                        break

                if not file_path:
                    return "Error: Failed to retrieve selected file path"

                try:
                    # Call json2db function and get task ID
                    task_id = json2db(file_path)
                    current_task_id = task_id  # Ensure update global variable

                    return f"File {selected_file} successfully stored to database!\nTask ID: {task_id}\nThis ID will be used for subsequent operation path understanding."
                except Exception as e:
                    import traceback

                    return (
                        f"Error storing to database: {str(e)}\n{traceback.format_exc()}"
                    )

            # Bind store to database button event
            store_to_db_btn.click(
                fn=store_to_database,
                inputs=[json_files],
                outputs=[evolution_log],
            )

            # Button click event
            def on_button_click(action_type, selected_file):
                if not selected_file:
                    return "Please select a file first"
                return f"{action_type} - Selected file: {selected_file}"

            # Define function to understand operation path
            def understand_chain(selected_file, progress=gr.Progress()):
                global current_files, current_task_id

                if not selected_file:
                    return "Error: Please select a file first"

                # Create variable to store current log status
                current_log = ["Starting to understand operation path..."]

                # Add log function to update UI
                def add_log(message):
                    current_log.append(message)
                    return "\n".join(current_log)

                # Check if task ID exists, if not try to generate from filename
                if not current_task_id:
                    add_log(
                        "Warning: Task ID not found, attempting to generate temporary task ID from filename..."
                    )
                    result = refresh_task_id_from_file(selected_file)
                    add_log(result)

                    if not current_task_id:
                        add_log(
                            "Error: Could not get task ID, cannot continue understanding operation path"
                        )
                        return "\n".join(current_log)

                add_log(f"Current task ID: {current_task_id}")
                progress(0.1, "Initializing database connection...")

                try:
                    # Connect to Neo4j database
                    db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH)

                    # Get all start nodes
                    add_log("Getting start nodes...")
                    progress(0.2, "Getting start nodes...")
                    start_nodes = db.get_chain_start_nodes()

                    if not start_nodes:
                        add_log("❌ No start nodes found")
                        return "\n".join(current_log)

                    add_log(f"Found {len(start_nodes)} start nodes")
                    progress(0.3, "Matching start nodes...")

                    # Find start node matching current task ID
                    matching_node = None
                    for idx, node in enumerate(start_nodes):
                        try:
                            if "other_info" in node:
                                other_info = node["other_info"]
                                if isinstance(other_info, str):
                                    other_info = json.loads(other_info)

                                    add_log(
                                        f"Checking node {idx+1}/{len(start_nodes)}: {node['page_id']}"
                                    )

                                    if (
                                        "task_info" in other_info
                                        and "task_id" in other_info["task_info"]
                                    ):
                                        node_task_id = other_info["task_info"][
                                            "task_id"
                                        ]
                                        add_log(f"  - Node task ID: {node_task_id}")

                                        # Check for match, including handling temporary IDs
                                        if node_task_id == current_task_id or (
                                            current_task_id.startswith("temp_")
                                            and node == start_nodes[-1]
                                        ):
                                            matching_node = node
                                            add_log(f"  - ✓ Found matching node!")
                                            break
                                    else:
                                        add_log("  - Node missing task ID information")
                            else:
                                add_log("  - Node missing other_info field")
                        except Exception as e:
                            add_log(f"  - Error parsing node information: {str(e)}")

                    # If no matching node found but using temporary ID, use first node
                    if (
                        not matching_node
                        and current_task_id.startswith("temp_")
                        and start_nodes
                    ):
                        matching_node = start_nodes[0]
                        add_log(
                            f"Using temporary ID, defaulting to first start node: {matching_node['page_id']}"
                        )

                    if not matching_node:
                        add_log(
                            f"❌ No start node found matching task ID {current_task_id}"
                        )
                        return "\n".join(current_log)

                    add_log(f"✓ Using start node: {matching_node['page_id']}")
                    progress(0.4, "Processing operation chain...")

                    # Use async function to process chain
                    async def process_chain():
                        nonlocal current_log
                        add_log("Starting to process operation chain...")
                        progress(0.5, "Analyzing operation chain...")

                        try:
                            add_log(
                                "Calling AI for chain analysis, this may take some time..."
                            )
                            processed_triplets = await process_and_update_chain(
                                matching_node["page_id"]
                            )
                            progress(0.8, "Analysis complete, processing results...")

                            if not processed_triplets:
                                add_log("❌ No processable triplets found")
                                return

                            add_log(
                                f"✓ Successfully processed {len(processed_triplets)} triplets"
                            )

                            # Show example of first triplet processing result
                            if processed_triplets:
                                first_triplet = processed_triplets[0]
                                add_log("\nExample triplet processing result:")

                                # Source page information
                                add_log(
                                    f"Source page ID: {first_triplet['source_page']['page_id']}"
                                )
                                description = first_triplet["source_page"].get(
                                    "description", "N/A"
                                )
                                add_log(
                                    f"Page description: {description[:150]}..."
                                    if len(description) > 150
                                    else f"Page description: {description}"
                                )

                                # Element information
                                add_log(
                                    f"Element ID: {first_triplet['element']['element_id']}"
                                )
                                ele_desc = first_triplet["element"].get(
                                    "description", "N/A"
                                )
                                add_log(
                                    f"Element description: {ele_desc[:150]}..."
                                    if len(ele_desc) > 150
                                    else f"Element description: {ele_desc}"
                                )

                                # Target page information
                                add_log(
                                    f"Target page ID: {first_triplet['target_page']['page_id']}"
                                )
                                target_desc = first_triplet["target_page"].get(
                                    "description", "N/A"
                                )
                                add_log(
                                    f"Target description: {target_desc[:150]}..."
                                    if len(target_desc) > 150
                                    else f"Target description: {target_desc}"
                                )

                                # Reasoning result summary
                                if "reasoning" in first_triplet:
                                    reasoning = first_triplet["reasoning"]
                                    add_log("\nReasoning Result Summary:")

                                    if "user_intent" in reasoning:
                                        user_intent = reasoning["user_intent"]
                                        add_log(
                                            f"User Intent: {user_intent[:150]}..."
                                            if len(user_intent) > 150
                                            else f"User Intent: {user_intent}"
                                        )

                                    if "context" in reasoning:
                                        context = reasoning["context"]
                                        add_log(
                                            f"Operation Context: {context[:150]}..."
                                            if len(context) > 150
                                            else f"Operation Context: {context}"
                                        )

                                    if "state_change" in reasoning:
                                        state_change = reasoning["state_change"]
                                        add_log(
                                            f"State Change: {state_change[:150]}..."
                                            if len(state_change) > 150
                                            else f"State Change: {state_change}"
                                        )
                        except Exception as e:
                            add_log(f"Error processing chain: {str(e)}")
                            import traceback

                            add_log(traceback.format_exc())

                        add_log("\n✨ Operation path understanding complete! ✨")
                        progress(1.0, "Complete!")

                    # Start background task
                    import asyncio
                    import time

                    # Create a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Run background task in background thread
                    bg_thread = threading.Thread(
                        target=lambda: loop.run_until_complete(process_chain())
                    )
                    bg_thread.start()

                    # Wait for task to complete and update UI in real-time
                    start_time = time.time()
                    while bg_thread.is_alive():
                        # Update every 500ms
                        time.sleep(0.5)
                        yield "\n".join(current_log)

                    # Ensure final update
                    yield "\n".join(current_log)

                except Exception as e:
                    add_log(f"❌ Error understanding operation path: {str(e)}")
                    import traceback

                    add_log(traceback.format_exc())
                    yield "\n".join(current_log)

            understand_btn.click(
                fn=understand_chain,
                inputs=[json_files],
                outputs=[evolution_log],
                queue=True,  # Enable queue processing
            )

            # Define function to generate advanced path
            def generate_high_level_action(selected_file, progress=gr.Progress()):
                global current_files, current_task_id

                if not selected_file:
                    return "Error: Please select a file first"

                # Create variable to store current log status
                current_log = ["Starting to generate advanced path..."]

                # Add log function to update UI
                def add_log(message):
                    current_log.append(message)
                    return "\n".join(current_log)

                # Check if task ID exists, if not try to generate from filename
                if not current_task_id:
                    add_log(
                        "Warning: Task ID not found, attempting to generate temporary task ID from filename..."
                    )
                    result = refresh_task_id_from_file(selected_file)
                    add_log(result)

                    if not current_task_id:
                        add_log(
                            "Error: Could not get task ID, cannot continue generating advanced path"
                        )
                        return "\n".join(current_log)

                add_log(f"Current task ID: {current_task_id}")
                progress(0.1, "Initializing database connection...")

                try:
                    # Connect to Neo4j database
                    db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH)

                    # Get all start nodes
                    add_log("Getting start nodes...")
                    progress(0.2, "Getting start nodes...")
                    start_nodes = db.get_chain_start_nodes()

                    if not start_nodes:
                        add_log("❌ No start nodes found")
                        return "\n".join(current_log)

                    add_log(f"Found {len(start_nodes)} start nodes")
                    progress(0.3, "Matching start nodes...")

                    # Find start node matching current task ID
                    matching_node = None
                    for idx, node in enumerate(start_nodes):
                        try:
                            if "other_info" in node:
                                other_info = node["other_info"]
                                if isinstance(other_info, str):
                                    other_info = json.loads(other_info)

                                    add_log(
                                        f"Checking node {idx+1}/{len(start_nodes)}: {node['page_id']}"
                                    )

                                    if (
                                        "task_info" in other_info
                                        and "task_id" in other_info["task_info"]
                                    ):
                                        node_task_id = other_info["task_info"][
                                            "task_id"
                                        ]
                                        add_log(f"  - Node task ID: {node_task_id}")

                                        # Check for match, including handling temporary IDs
                                        if node_task_id == current_task_id or (
                                            current_task_id.startswith("temp_")
                                            and node == start_nodes[-1]
                                        ):
                                            matching_node = node
                                            add_log(f"  - ✓ Found matching node!")
                                            break
                                    else:
                                        add_log("  - Node missing task ID information")
                            else:
                                add_log("  - Node missing other_info field")
                        except Exception as e:
                            add_log(f"  - Error parsing node information: {str(e)}")

                    # If no matching node found but using temporary ID, use first node
                    if (
                        not matching_node
                        and current_task_id.startswith("temp_")
                        and start_nodes
                    ):
                        matching_node = start_nodes[0]
                        add_log(
                            f"Using temporary ID, defaulting to first start node: {matching_node['page_id']}"
                        )

                    if not matching_node:
                        add_log(
                            f"❌ No start node found matching task ID {current_task_id}"
                        )
                        return "\n".join(current_log)

                    add_log(f"✓ Using start node: {matching_node['page_id']}")
                    progress(0.4, "Evaluating chain templatability...")

                    # Use async function to process chain evolution
                    async def process_evolution():
                        nonlocal current_log
                        add_log("Starting chain evolution process...")
                        progress(0.5, "Evaluating chain...")

                        try:
                            # Redirect standard output to capture evolve_chain_to_action output
                            import io
                            import sys
                            from contextlib import redirect_stdout

                            # Call evolve_chain_to_action function
                            add_log(
                                "Evaluating chain templatability, this may take some time..."
                            )
                            progress(0.6, "Evaluating chain...")

                            # Use StringIO to capture output
                            f = io.StringIO()
                            with redirect_stdout(f):
                                # Call chain evolution function
                                action_id = await evolve_chain_to_action(
                                    matching_node["page_id"]
                                )

                            # Get captured output and add to log
                            output = f.getvalue()
                            for line in output.splitlines():
                                if line.strip():  # Ignore empty lines
                                    add_log(line)

                            progress(
                                0.9, "Chain evolution completed, processing results..."
                            )

                            # Process return result
                            if action_id:
                                add_log(
                                    f"✓ Successfully created advanced action node, ID: {action_id}"
                                )
                                # Query database for more advanced action node details
                                add_log("Getting advanced action node details...")

                                # Here, you can add code to query advanced action node details
                                # For example: action_node = db.get_action_node(action_id)

                                add_log(
                                    "Advanced path generation completed successfully!"
                                )
                            else:
                                add_log(
                                    "❌ Chain evolution failed, unable to create advanced action node"
                                )
                                add_log(
                                    "Possible reasons: Chain evaluation as not templatable, or error occurred during generation"
                                )

                        except Exception as e:
                            add_log(f"Chain evolution error: {str(e)}")
                            import traceback

                            add_log(traceback.format_exc())

                        add_log("\n✨ Chain evolution processing completed! ✨")
                        progress(1.0, "Complete!")

                    # Start background task
                    import asyncio
                    import time

                    # Create a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Run background task in background thread
                    bg_thread = threading.Thread(
                        target=lambda: loop.run_until_complete(process_evolution())
                    )
                    bg_thread.start()

                    # Wait for task to complete and update UI in real-time
                    start_time = time.time()
                    while bg_thread.is_alive():
                        # Update every 500ms
                        time.sleep(0.5)
                        yield "\n".join(current_log)

                    # Ensure final update
                    yield "\n".join(current_log)

                except Exception as e:
                    add_log(f"❌ Error generating advanced path: {str(e)}")
                    import traceback

                    add_log(traceback.format_exc())
                    yield "\n".join(current_log)

            # Modify generate advanced path button binding
            generate_btn.click(
                fn=generate_high_level_action,
                inputs=[json_files],
                outputs=[evolution_log],
                queue=True,  # Enable queue processing
            )

            # Tab 5: Action Execution
            with gr.Tab("Action Execution"):
                gr.Markdown("### Action Execution Management")
                with gr.Row():
                    with gr.Column(scale=1):
                        device_selector_exec = gr.Radio(
                            label="Select Execution Device",
                            choices=get_adb_devices(),  # Assumes get_adb_devices is defined
                            interactive=True,
                        )
                        refresh_device_btn = gr.Button(
                            "Refresh Device List", variant="secondary"
                        )
                        task_input = gr.Textbox(
                            label="Enter Task Description",
                            placeholder="e.g., Open settings and switch to airplane mode",
                            lines=3,
                        )
                        with gr.Row():
                            execute_btn = gr.Button(
                                "Execute Task", variant="primary", scale=2
                            )
                            stop_btn = gr.Button(
                                "Stop Execution", variant="stop", scale=1, interactive=False
                            )
                        execution_status = gr.Textbox(
                            label="Execution Status", interactive=False
                        )
                    with gr.Column(scale=2):
                        execution_log = gr.TextArea(
                            label="Execution Log", interactive=False, lines=15
                        )
                        screenshot_gallery_exec = gr.Gallery(
                            label="Execution Process Screenshots", columns=2, height=400
                        )

                # Import the deployment run_task function correctly once for this tab's scope
                try:
                    from .deployment import run_task as deployment_run_task_entry_point

                    print("DEMO.PY (Action Execution Tab): Successfully imported .deployment.run_task")
                except ImportError as e_import:
                    print(f"DEMO.PY (Action Execution Tab): FAILED to import .deployment.run_task. Error: {e_import}")


                    # Define a placeholder so Gradio doesn't crash on load if import fails
                    def deployment_run_task_entry_point(*args, **kwargs):
                        raise ImportError("Critical: .deployment.run_task could not be imported for Action Execution.")


                def update_execution_devices():
                    devices = get_adb_devices()
                    return gr.update(choices=devices)


                refresh_device_btn.click(
                    fn=update_execution_devices, outputs=[device_selector_exec]
                )


                def execute_deployment_task(
                        selected_device_id_str: str, task_description_str: str, progress=gr.Progress()
                ):
                    if not selected_device_id_str or selected_device_id_str == "No devices found":
                        return "Error: Please select a valid device", "Execution failed: No valid device selected", []
                    if not task_description_str or task_description_str.strip() == "":
                        return "Error: Please enter task description", "Execution failed: Task description is empty", []

                    logs = []
                    screenshots_for_ui = []  # For displaying in Gradio gallery for this tab
                    action_execution_queue = Queue()

                    def add_log_entry(message: str):
                        logs.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

                    def task_execution_thread_target():
                        try:
                            add_log_entry(
                                f"Thread: Starting deployment task process for: '{task_description_str}' on device '{selected_device_id_str}'")

                            # Call the imported deployment function
                            result_dictionary = deployment_run_task_entry_point(
                                task=task_description_str,
                                device=selected_device_id_str
                            )
                            action_execution_queue.put(result_dictionary)
                            add_log_entry("Thread: Deployment task function call completed.")

                        except ImportError as ie_thread:  # Should be caught by top-level import ideally
                            add_log_entry(f"Thread Import Error: {str(ie_thread)}")
                            action_execution_queue.put(
                                {"status": "error", "message": f"ImportError in thread: {str(ie_thread)}",
                                 "completed": False})
                        except Exception as e_thread:
                            tb_str = traceback.format_exc()
                            add_log_entry(f"Thread Execution Error: {str(e_thread)}\n{tb_str}")
                            action_execution_queue.put(
                                {"status": "error", "message": str(e_thread), "completed": False})

                    add_log_entry(
                        f"UI: Kicking off task: '{task_description_str}' on device '{selected_device_id_str}'.")
                    progress(0.1, "Initializing execution environment...")

                    execution_thread = threading.Thread(target=task_execution_thread_target)
                    execution_thread.start()

                    current_ui_status = "Executing..."
                    loop_start_time = time.time()

                    while execution_thread.is_alive() or not action_execution_queue.empty():
                        try:
                            # Try to get a result if the thread might have put something
                            # This is primarily for the final result. Intermediate results would require
                            # deployment_run_task_entry_point to put multiple items on the queue.
                            if not execution_thread.is_alive() and not action_execution_queue.empty():
                                final_result_payload = action_execution_queue.get_nowait()
                                action_execution_queue.put(final_result_payload)  # Put back for final processing
                                break  # Exit UI update loop, proceed to final result processing

                            # Update UI periodically
                            yield "\n".join(logs), current_ui_status, screenshots_for_ui
                            time.sleep(0.5)  # UI refresh rate
                            progress(((time.time() - loop_start_time) / 30) % 0.8 + 0.1,
                                     current_ui_status)  # Example progress update

                        except Empty:
                            if not execution_thread.is_alive():  # Thread finished and queue became empty
                                break
                            # Queue is empty, but thread is still running, continue waiting
                        except Exception as e_loop_update:
                            add_log_entry(f"UI Update Loop Error: {str(e_loop_update)}")
                            current_ui_status = "Error in UI update loop"
                            yield "\n".join(logs), current_ui_status, screenshots_for_ui
                            break  # Exit loop on UI error

                    # Process final result from the queue
                    final_payload_from_thread = None
                    try:
                        final_payload_from_thread = action_execution_queue.get(
                            timeout=10.0)  # Wait up to 10s for final result
                    except Empty:
                        add_log_entry(
                            "❓ UI: Execution thread ended, but no final result obtained from queue (timeout).")
                        current_ui_status = "Execution ended (timeout waiting for result)."
                    except Exception as e_q_final:
                        add_log_entry(f"❓ UI: Error getting final result from queue: {str(e_q_final)}")
                        current_ui_status = "Error getting final result."

                    if final_payload_from_thread:
                        status = final_payload_from_thread.get("status", "unknown")
                        message = final_payload_from_thread.get("message", "No specific message.")
                        completed_status_flag = final_payload_from_thread.get("completed", False)

                        if status == "success":
                            current_ui_status = "✅ Execution successful: Task completed by deployment." if completed_status_flag else "⚠️ Deployment process successful but task not marked as fully completed."
                        else:
                            current_ui_status = f"❌ Execution failed by deployment: {message}"
                        add_log_entry(current_ui_status)  # Log the final status derived from payload

                    progress(1.0, "Execution processing finished.")
                    add_log_entry("UI: Execution control process completed.")
                    return "\n".join(logs), current_ui_status, screenshots_for_ui


                execute_btn.click(
                    fn=execute_deployment_task,
                    inputs=[device_selector_exec, task_input],
                    outputs=[execution_log, execution_status, screenshot_gallery_exec],
                    queue=True,
                )


                def toggle_execution_buttons(is_executing=True):
                    return gr.update(interactive=not is_executing), gr.update(interactive=is_executing)


                execute_btn.click(
                    fn=lambda: toggle_execution_buttons(True),
                    outputs=[execute_btn, stop_btn],
                )
                stop_btn.click(  # Note: Stop functionality is not fully implemented here, just UI toggle
                    fn=lambda: toggle_execution_buttons(False),
                    outputs=[execute_btn, stop_btn],
                )
                demo.load(fn=update_execution_devices, outputs=[device_selector_exec])

if __name__ == "__main__":
    demo.launch()
