from typing import List, Dict, Optional, Annotated, Callable, Any
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class State(TypedDict):
    """
    State machine for learning mode
    """

    # Task related
    tsk: str  # Current task name (e.g., open an app, find specific content)
    app_name: str  # Current application name
    completed: bool  # Whether the task is completed
    step: int  # Counter for current execution step
    history_steps: List[
        Dict
    ]  # Historical step records (detailed operation data for each step)

    # Page related
    page_history: List[
        str
    ]  # Historical page snapshot data (including screenshots, elements, etc.)
    current_page_screenshot: Optional[
        str
    ]  # Path or binary data of current page screenshot
    current_page_json: Optional[str]  # Parsed JSON data of current page

    # Operation related
    recommend_action: str  # Currently recommended action
    clicked_elements: List[
        Dict
    ]  # List of clicked elements, each can include position, text, ID, etc.
    action_reflection: List[Dict]  # Reflection content for each operation
    tool_results: List[
        Dict
    ]  # Results of tool calls (e.g., returns from OCR, API calls, etc.)

    # Device related
    device: str  # Device name or ID
    device_info: Dict  # Detailed device information (resolution, etc.)

    # Context related
    context: Annotated[
        list, add_messages
    ]  # Current task context, containing messages or steps to execute
    errors: List[
        Dict
    ]  # Error records, containing error information for steps, pages, actions, etc.

    callback: Optional[
        Callable[[TypedDict], None]
    ]  # Callback function, accepts current State and returns None


class DeploymentState(TypedDict):
    """
    State machine for task deployment execution
    """

    # Task related
    task: str  # User input task description
    completed: bool  # Whether the task is completed
    current_step: int  # Current execution step index
    total_steps: int  # Total number of steps
    execution_status: str  # Execution status (ready/running/success/error)
    retry_count: int  # Current retry count
    max_retries: int  # Maximum retry count

    # Device related
    device: str  # Device ID

    # Page information
    current_page: (
        Dict  # Current page information, including screenshot path and element data
    )

    # Execution related
    current_element: Optional[Dict]  # Current element being operated
    current_action: Optional[Dict]  # Current high-level action being executed
    matched_elements: List[Dict]  # List of matched screen elements
    associated_shortcuts: List[Dict]  # List of associated shortcuts
    execution_template: Optional[Dict]  # Execution template

    # Records and messages
    history: List[Dict]  # Execution history records
    messages: Annotated[list, add_messages]  # Message history for React mode

    # Execution flow control
    should_fallback: bool  # Whether to fall back to basic operation mode
    should_execute_shortcut: bool  # Whether to execute shortcut operations

    # Callback
    callback: Optional[Callable[[TypedDict], None]]  # Callback function


def create_deployment_state(
    task: str,
    device: str,
    max_retries: int = 3,
    callback: Optional[Callable[[TypedDict], None]] = None,
) -> DeploymentState:
    """
    Create and initialize DeploymentState object

    Args:
        task: User input task description
        device: Device ID
        max_retries: Maximum retry count, default is 3
        callback: Callback function (optional)

    Returns:
        Initialized DeploymentState object
    """
    # Basic default values
    state: Dict[str, Any] = {}

    # Initialize all fields, ensure all fields have default values
    # Task related
    state["task"] = task
    state["completed"] = False
    state["current_step"] = 0
    state["total_steps"] = 0
    state["execution_status"] = "ready"
    state["retry_count"] = 0
    state["max_retries"] = max_retries

    # Device related
    state["device"] = device

    # Page information
    state["current_page"] = {
        "screenshot": None,
        "elements_json": None,
        "elements_data": [],
    }

    # Execution related
    state["current_element"] = None
    state["current_action"] = None
    state["matched_elements"] = []
    state["associated_shortcuts"] = []
    state["execution_template"] = None

    # Records and messages
    state["history"] = []
    state["messages"] = []

    # Execution flow control
    state["should_fallback"] = False
    state["should_execute_shortcut"] = False

    # Callback
    state["callback"] = callback

    # Check if all fields are initialized
    for key in DeploymentState.__annotations__:
        if key not in state:
            raise ValueError(f"State initialization failed: missing field '{key}'")

    return state


class ActionMatch(BaseModel):
    action_id: str = Field(description="High-level action node ID")
    name: str = Field(description="High-level action name")
    match_score: float = Field(description="Match score")
    reason: str = Field(description="Match reason explanation")


class ElementMatch(BaseModel):
    element_id: str = Field(description="Element ID")
    match_score: float = Field(description="Match score")
    screen_element_id: int = Field(description="Screen element ID")
    action_type: str = Field(description="Atomic operation type")
    parameters: Dict[str, Any] = Field(description="Operation parameters")
