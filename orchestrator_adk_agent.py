# orchestrator_adk_agent.py
import os
import json
import asyncio
import sys
from datetime import datetime, \
    timezone  # Not directly used in this refactored version, but good to keep if tools evolve
from typing import Dict, Any, List, Optional
import uuid
import logging


logger_orch = logging.getLogger(__name__)
if __name__ == "__main__":  # When run directly for testing this module's components
    if not logging.getLogger().hasHandlers():  # Check if root logger is already configured
        logging.basicConfig(level=os.getenv("LOG_LEVEL_ORCH", "DEBUG").upper(),
                            format='%(asctime)s - ORCH_TEST - %(name)s - %(levelname)s - %(message)s')

# --- Dynamically add project root to sys.path ---
SCRIPT_DIR_ORCHESTRATOR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR_ORCHESTRATOR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_ORCHESTRATOR)
# logger_orch.info(f"Orchestrator: Added SCRIPT_DIR '{SCRIPT_DIR_ORCHESTRATOR}' to sys.path.") # Can be verbose on import

from dotenv import load_dotenv

# --- ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService, Session as ADKSession
from google.adk.runners import Runner
from google.genai.types import Content as ADKContent, Part as ADKPart

# --- Aura System Imports ---
from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_connector import MemoryConnector

# --- Load .env file from project root ---
# This will be loaded when a2a_wrapper/main.py runs, but good to have if testing this module directly
dotenv_path_orch = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')  # Use a distinct variable
if os.path.exists(dotenv_path_orch):
    logger_orch.debug(f"Orchestrator Module: Loading .env file from: {dotenv_path_orch}")
    load_dotenv(dotenv_path_orch, override=False)  # override=False so main.py's load_dotenv takes precedence
else:
    logger_orch.debug(f"Orchestrator Module: .env file not found at {dotenv_path_orch}.")

# --- Configuration (Exported for a2a_wrapper/main.py) ---
AGENT_MODEL_STRING = os.getenv("AURA_MODEL_IDENTIFIER", "openrouter/openai/gpt-4o-mini")
AGENT_MODEL = LiteLlm(model=AGENT_MODEL_STRING)
ADK_APP_NAME = "AuraNCFOrchestratorApp_AppAgentX_Phase2"  # Exported
logger_orch.info(f"Orchestrator Module: ADK App Name='{ADK_APP_NAME}', Agent Model='{AGENT_MODEL_STRING}'")

# --- Initialize MemoryBlossom (Exported) ---
memory_blossom_persistence_file = os.getenv("MEMORY_BLOSSOM_PERSISTENCE_PATH",
                                            "memory_blossom_orchestrator_default.json")  # Potentially different default if needed
memory_blossom_instance = MemoryBlossom(persistence_path=memory_blossom_persistence_file)  # Exported
memory_connector_instance = MemoryConnector(memory_blossom_instance)
memory_blossom_instance.set_memory_connector(memory_connector_instance)
logger_orch.info(f"Orchestrator Module: MemoryBlossom initialized. Persistence: '{memory_blossom_persistence_file}'")

# --- AppAgentX Integration (TRY-EXCEPT BLOCK REMOVED FOR DEBUGGING) ---
APPAGENTX_DEPLOYMENT_AVAILABLE = False # Initialize to False
APPAGENTX_LEARNING_AVAILABLE = False   # Initialize to False
appagentx_config_module = None         # Initialize to None
APPAGENTX_NEO4J_URI = None             # Initialize to None
APPAGENTX_NEO4J_AUTH = None            # Initialize to None

logger_orch.info("Orchestrator Module: Attempting to import AppAgentX modules (try-except removed for debug)...")
# --- DIRECT IMPORTS: If any of these fail, the script will crash here, showing a full traceback ---
from AppAgentX.deployment import run_task as appagentx_run_task_deployment
import AppAgentX.config as appagentx_config_module # This will re-assign if successful
from AppAgentX.explor_auto import run_task as appagentx_explore_task_auto
from AppAgentX.chain_evolve import evolve_chain_to_action
from AppAgentX.data.data_storage import state2json as appagentx_state2json, json2db as appagentx_json2db
from AppAgentX.data.graph_db import Neo4jDatabase as AppAgentXNeo4jDatabase
from AppAgentX.data.State import State as AppAgentXInternalState # Make sure this specific import path is what you need
from AppAgentX.tool.screen_content import get_device_size as appagentx_get_device_size
# --- END DIRECT IMPORTS ---

# If all imports above succeed, then AppAgentX is considered available:
APPAGENTX_DEPLOYMENT_AVAILABLE = True
APPAGENTX_LEARNING_AVAILABLE = True
APPAGENTX_NEO4J_URI = getattr(appagentx_config_module, "Neo4j_URI", None) # Get from successfully imported module
APPAGENTX_NEO4J_AUTH = getattr(appagentx_config_module, "Neo4j_AUTH", None)
logger_orch.info("Orchestrator Module: Successfully imported ALL required AppAgentX modules directly.")


# --- ADK Tools (Defined here, used by LlmAgent) ---
def add_memory_tool_func(
        content: str, memory_type: str, emotion_score: float = 0.0, coherence_score: float = 0.5,
        novelty_score: float = 0.5, initial_salience: float = 0.5, metadata_json: Optional[str] = None,
        tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    logger_orch.info(f"TOOL (Orch): add_memory - Type='{memory_type}', Content='{content[:30]}...'")
    parsed_metadata = {}
    if metadata_json:
        try:
            parsed_metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            logger_orch.warning(f"TOOL (Orch): Invalid JSON for metadata in add_memory: {metadata_json}")
            return {"status": "error", "message": "Invalid JSON format for metadata."}

    parsed_metadata.setdefault('source', 'aura_agent_tool_via_orchestrator_module')
    if tool_context:
        if tool_context.user_id: parsed_metadata['user_id'] = tool_context.user_id
        if tool_context.session_id: parsed_metadata['session_id'] = tool_context.session_id
    try:
        memory = memory_blossom_instance.add_memory(
            content=content, memory_type=memory_type, metadata=parsed_metadata,
            emotion_score=emotion_score, coherence_score=coherence_score,
            novelty_score=novelty_score, initial_salience=initial_salience
        )
        memory_blossom_instance.save_memories()  # Save after each add
        logger_orch.debug(f"TOOL (Orch): Memory added successfully: ID={memory.id}")
        return {"status": "success", "memory_id": memory.id,
                "message": f"Memory of type '{memory_type}' added."}
    except Exception as e:
        logger_orch.error(f"TOOL (Orch): Error in add_memory_tool_func: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def reflector_add_memory(  # This is also a function that can be imported and used
        content: str, memory_type: str, emotion_score: float = 0.0,
        coherence_score: float = 0.5, novelty_score: float = 0.5,  # these scores are not in MemoryBlossom.add_memory
        initial_salience: float = 0.5, metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logger_orch.info(f"REFLECTOR (Orch): Adding memory - Type='{memory_type}', Content='{content[:30]}...'")
    try:
        final_metadata = metadata.copy() if metadata else {}
        final_metadata.setdefault('source', 'reflector_add_memory_orchestrator_module')
        # Assuming MemoryBlossom.add_memory is the target; it doesn't use coherence/novelty directly
        memory = memory_blossom_instance.add_memory(
            content=content, memory_type=memory_type, metadata=final_metadata,
            emotion_score=emotion_score, initial_salience=initial_salience
            # coherence_score and novelty_score are not standard params for MemoryBlossom.add_memory
        )
        memory_blossom_instance.save_memories()
        logger_orch.debug(f"REFLECTOR (Orch): Memory added successfully: ID={memory.id}")
        return {"status": "success", "memory_id": memory.id,
                "message": f"Reflector added memory of type '{memory_type}'."}
    except Exception as e:
        logger_orch.error(f"REFLECTOR (Orch): Error in reflector_add_memory: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def recall_memories_tool_func(
        query: str, target_memory_types_json: Optional[str] = None, top_k: int = 3,
        tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    logger_orch.info(f"TOOL (Orch): recall_memories - Query='{query[:30]}...'")
    target_types_list: Optional[List[str]] = None
    if target_memory_types_json:
        try:
            target_types_list = json.loads(target_memory_types_json)
            if not isinstance(target_types_list, list) or not all(isinstance(item, str) for item in target_types_list):
                logger_orch.warning("TOOL (Orch): target_memory_types_json was not a list of strings.")
                return {"status": "error",
                        "message": "target_memory_types_json must be a JSON string of a list of strings."}
        except json.JSONDecodeError:
            logger_orch.warning(f"TOOL (Orch): Invalid JSON for target_memory_types_json: {target_memory_types_json}")
            return {"status": "error", "message": "Invalid JSON format for target_memory_types_json."}
    try:
        conversation_history = None
        if tool_context and tool_context.state and isinstance(tool_context.state, dict):
            conversation_history = tool_context.state.get('conversation_history', [])

        recalled_memories = memory_blossom_instance.retrieve_memories(
            query=query, target_memory_types=target_types_list, top_k=top_k,
            conversation_context=conversation_history, apply_criticality=True
        )
        logger_orch.debug(f"TOOL (Orch): Recalled {len(recalled_memories)} memories.")
        return {
            "status": "success", "count": len(recalled_memories),
            "memories": [mem.to_dict() for mem in recalled_memories]
        }
    except Exception as e:
        logger_orch.error(f"TOOL (Orch): Error in recall_memories_tool_func: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def execute_android_task_tool_func(
        task_description: str, target_application: Optional[str] = None, device_id: str = "DYONHQPZ9PF6V4TO",
        tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    print(f"--- TOOL: execute_android_task_tool_func called ---")
    print(f"  Task: {task_description}")
    if target_application: print(f"  Target App: {target_application}")
    print(f"  Device: {device_id}")

    if not APPAGENTX_DEPLOYMENT_AVAILABLE:
        return {"status": "error", "message": "AppAgentX deployment functionality is not available."}

    full_task_for_appagentx = task_description
    if target_application:
        full_task_for_appagentx = f"In the app '{target_application}', {task_description}"
    try:
        print(f"  Calling AppAgentX with task: '{full_task_for_appagentx}' on device '{device_id}'")
        result = appagentx_run_task_deployment(task=full_task_for_appagentx, device=device_id)
        logger_orch.info(f"TOOL execute_android_task: Received task_description='{task_description}'")
        logger_orch.info(f"TOOL execute_android_task: Received target_application='{target_application}'")
        print(f"  AppAgentX Result: {result}")
        if result.get("status") == "success" or result.get("execution_status") == "success":
            return {
                "status": "success",
                "message": result.get("message", "Task execution initiated or completed by AppAgentX."),
                "details": {"steps_completed": result.get("steps_completed", result.get("current_step", "N/A")),
                            "total_steps_planned": result.get("total_steps", "N/A")}
            }
        else:
            return {"status": "error",
                    "message": result.get("message", "AppAgentX reported an error or did not complete successfully."),
                    "error_details": result.get("error", result.get("error_details",
                                                                    "No specific error detail from AppAgentX."))}
    except Exception as e:
        print(f"Error calling AppAgentX deployment: {str(e)}")
        import traceback;
        traceback.print_exc()
        return {"status": "error", "message": f"Failed to execute Android task via AppAgentX: {str(e)}"}


execute_android_task_adk_tool = FunctionTool(func=execute_android_task_tool_func)


async def learn_android_procedure_tool_func(
        task_to_learn: str,
        target_application: Optional[str] = None,
        device_id: str = "DYONHQPZ9PF6V4TO",
        tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Instructs AppAgentX to explore and learn a new procedure for the given task on an Android device.
    The learned procedure (a high-level action) will be stored in AppAgentX's database and a summary in Aura's memory.
    Args:
        task_to_learn (str): A clear description of the new task or procedure to learn.
        target_application (str, optional): The primary application involved in the task.
        device_id (str, optional): The Android device ID to use for exploration.
        tool_context (ToolContext, optional): ADK ToolContext.
    Returns:
        dict: A dictionary indicating the status ('success' or 'error') and a message.
    """
    print(f"--- TOOL: learn_android_procedure_tool_func called ---")
    print(f"  Task to learn: {task_to_learn}")
    if target_application: print(f"  Target App: {target_application}")

    if not APPAGENTX_LEARNING_AVAILABLE:
        return {"status": "error", "message": "AppAgentX learning functionality is not available."}

    # 1. Prepare AppAgentX's initial state for exploration
    try:
        print("  Initializing AppAgentX state for exploration...")
        # This needs AppAgentX's get_device_size tool and State class
        device_size_info = appagentx_get_device_size.invoke({"device": device_id})
        if isinstance(device_size_info, str) and "Error" in device_size_info:  # Check if get_device_size failed
            return {"status": "error", "message": f"Failed to get device size for AppAgentX: {device_size_info}"}

        initial_appagentx_state_dict = {
            "tsk": task_to_learn,
            "app_name": target_application or "",
            "completed": False, "step": 0, "history_steps": [], "page_history": [],
            "current_page_screenshot": None, "recommend_action": "", "clicked_elements": [],
            "action_reflection": [], "tool_results": [], "device": device_id,
            "device_info": device_size_info, "context": [], "errors": [],
            "current_page_json": None, "callback": None,
        }
        # Convert dict to AppAgentXInternalState if necessary, or pass dict if its run_task accepts it
        # The `explor_auto.run_task` in AppAgentX.txt takes a `State` (TypedDict).
        # So, we directly pass the dictionary.
        print(f"  Initial AppAgentX state prepared: {task_to_learn[:50]}...")
    except Exception as e:
        return {"status": "error", "message": f"Failed to initialize AppAgentX state: {str(e)}"}

    # 2. Run AppAgentX exploration
    print("  Starting AppAgentX exploration...")
    try:
        # AppAgentX's explor_auto.run_task is NOT async.
        # To call a synchronous function from an async tool, use asyncio.to_thread
        loop = asyncio.get_running_loop()
        final_exploration_state_dict = await loop.run_in_executor(
            None, appagentx_explore_task_auto, initial_appagentx_state_dict
        )

        if not final_exploration_state_dict or final_exploration_state_dict.get(
                "status") == "error" or not final_exploration_state_dict.get(
            "history_steps"):  # check if exploration produced history
            msg = f"AppAgentX exploration failed or returned empty/error state: {final_exploration_state_dict.get('message', 'No history steps found in exploration result.')}"
            print(f"  {msg}")
            return {"status": "error", "message": msg}
        print("  AppAgentX exploration completed.")

        # Save exploration log using AppAgentX's state2json
        # Ensure the log directory exists (AppAgentX/log/json_state/)
        log_dir = os.path.join(os.path.dirname(appagentx_config_module.__file__), "..", "log", "json_state")
        os.makedirs(log_dir, exist_ok=True)

        # state2json might need a specific path or generates one.
        # Let's ensure it generates one based on timestamp if path is None.
        exploration_log_path = appagentx_state2json(final_exploration_state_dict)  # Pass the dict directly
        print(f"  AppAgentX exploration log saved to: {exploration_log_path}")

        # 3. Store exploration to AppAgentX's Neo4j DB
        print(f"  Storing exploration from '{exploration_log_path}' to AppAgentX Neo4j...")
        appagentx_task_id = appagentx_json2db(exploration_log_path)
        if not appagentx_task_id:
            return {"status": "error", "message": "Failed to store AppAgentX exploration log to its database."}
        print(f"  Exploration stored in AppAgentX DB with task ID: {appagentx_task_id}")

        # 4. Trigger chain evolution in AppAgentX
        # This requires identifying the start_page_id from the exploration.
        # We'll use the AppAgentX task_id to find the relevant start page.
        start_page_id_for_evolution = None
        try:
            appagentx_db = AppAgentXNeo4jDatabase(APPAGENTX_NEO4J_URI, APPAGENTX_NEO4J_AUTH)
            start_nodes = appagentx_db.get_chain_start_nodes()  # Gets all start nodes

            # Try to find the start node associated with appagentx_task_id
            for node in start_nodes:
                other_info_str = node.get("other_info", "{}")
                if isinstance(other_info_str, str):
                    try:
                        other_info = json.loads(other_info_str)
                        if other_info.get("task_info", {}).get("task_id") == appagentx_task_id:
                            start_page_id_for_evolution = node["page_id"]
                            break
                    except json.JSONDecodeError:
                        continue  # Malformed JSON in other_info
            appagentx_db.close()
        except Exception as db_e:
            print(f"  Error querying AppAgentX Neo4j for start_page_id: {db_e}")
            return {"status": "error", "message": f"Could not query AppAgentX DB for start page: {db_e}"}

        if not start_page_id_for_evolution:
            msg = f"Could not determine start_page_id for AppAgentX chain evolution for task_id '{appagentx_task_id}'. Evolution might use a default or fail."
            print(f"  WARNING: {msg}")
            # As a fallback, if AppAgentX's evolve_chain_to_action can take a task_id or find the latest,
            # that would be better. For now, we proceed cautiously or could error out.
            # Let's make it error out if no specific start_page_id is found for this task.
            return {"status": "error", "message": msg}

        print(f"  Evolving chain in AppAgentX starting from page ID: {start_page_id_for_evolution}...")
        learned_action_id = await evolve_chain_to_action(start_page_id_for_evolution)  # This is async

        if learned_action_id:
            success_message = f"Successfully learned new Android procedure via AppAgentX. Learned Action ID: {learned_action_id}"
            print(f"  {success_message}")

            # 5. Store a summary/reference in Aura's MemoryBlossom
            learned_action_details = None
            try:
                appagentx_db = AppAgentXNeo4jDatabase(APPAGENTX_NEO4J_URI, APPAGENTX_NEO4J_AUTH)
                learned_action_details = appagentx_db.get_action_by_id(learned_action_id)
                appagentx_db.close()
            except Exception as db_e_details:
                print(
                    f"  Warning: Could not retrieve details for learned action {learned_action_id} from AppAgentX DB: {db_e_details}")

            if learned_action_details:
                memory_content = (f"Learned Android procedure for task '{task_to_learn}': "
                                  f"{learned_action_details.get('name', 'Unnamed Procedure')}. "
                                  f"Description: {learned_action_details.get('description', 'No AppAgentX description.')}")
                add_memory_tool_func(  # Calling directly, not via LLM
                    content=memory_content,
                    memory_type="Procedural",
                    metadata_json=json.dumps({  # Pass metadata as JSON string
                        "source": "appagentx_learning_tool_phase2",
                        "appagentx_action_id": learned_action_id,
                        "task_learned": task_to_learn,
                        "target_application": target_application or "Unknown"
                    }),
                    initial_salience=0.85,  # Learned procedures are quite important
                    tool_context=tool_context  # Pass context for user_id/session_id if available
                )
                return {"status": "success", "message": success_message,
                        "learned_action_id": learned_action_id,
                        "procedure_name": learned_action_details.get('name', 'Unnamed Procedure')}
            else:
                return {"status": "warning",
                        "message": f"Procedure learned (ID: {learned_action_id}) but could not retrieve details for Aura's memory."}
        else:
            msg = "AppAgentX exploration completed and stored, but failed to evolve a new high-level action."
            print(f"  {msg}")
            return {"status": "error", "message": msg}

    except Exception as e:
        print(f"Error during AppAgentX learning process: {str(e)}")
        import traceback;
        traceback.print_exc()
        return {"status": "error", "message": f"Failed to learn Android procedure: {str(e)}"}


add_memory_adk_tool = FunctionTool(func=add_memory_tool_func)
recall_memories_adk_tool = FunctionTool(func=recall_memories_tool_func)
execute_android_task_adk_tool = FunctionTool(func=execute_android_task_tool_func)
learn_android_procedure_adk_tool = FunctionTool(func=learn_android_procedure_tool_func)

# --- Orchestrator ADK Agent Definition (Exported) ---
aura_agent_instruction = """
You are Aura, a helpful and insightful AI assistant with advanced memory and the ability to control Android applications and learn new Android procedures.
When asked to perform an action or learn something on Android, use the provided tools.
When adding memories, be specific about the memory_type.
"""
orchestrator_adk_agent_aura = LlmAgent(  # This is imported by a2a_wrapper
    name="AuraNCFOrchestrator_AppAgentX_Modular",  # Renamed for clarity
    model=AGENT_MODEL,
    instruction=aura_agent_instruction,
    tools=[
        add_memory_adk_tool,
        recall_memories_adk_tool,
        execute_android_task_adk_tool,
        learn_android_procedure_adk_tool
    ],
)

# --- ADK Session Service and Runner (Exported) ---
adk_session_service = InMemorySessionService()  # Exported and used by a2a_wrapper
adk_runner = Runner(  # Exported and used by a2a_wrapper
    agent=orchestrator_adk_agent_aura,
    app_name=ADK_APP_NAME,
    session_service=adk_session_service
)


# --- Main Execution Block (for testing this module's components ONLY) ---
async def run_orchestrator_module_test_conversation():
    user_id_test = "test_user_orchestrator_module"
    session_id_test = f"test_session_orch_module_{str(uuid.uuid4())[:4]}"
    logger_orch.info(f"\n--- Running Orchestrator Module (ADK Agent) Test Conversation ---")

    test_prompts = [
        "Hello Aura, can you add this memory: 'The sky is blue today.' as an Explicit memory?",
        "Aura, please recall memories about 'sky'."
    ]
    if APPAGENTX_DEPLOYMENT_AVAILABLE:
        test_prompts.append("Aura, try to open the calculator app.")
    if APPAGENTX_LEARNING_AVAILABLE:
        test_prompts.append("Aura, learn how to check the battery level on Android.")

    for test_prompt in test_prompts:
        adk_input_content = ADKContent(role="user", parts=[ADKPart(text=test_prompt)])
        logger_orch.info(f"\n>>> USER (Orch Test): {test_prompt}")
        final_response_text = None
        async for event in adk_runner.run_async(user_id=user_id_test, session_id=session_id_test,
                                                new_message=adk_input_content):
            if event.author == orchestrator_adk_agent_aura.name:
                if event.get_function_calls():
                    fc = event.get_function_calls()[0]
                    logger_orch.info(f"    ADK FC (Orch Test): {fc.name}({json.dumps(fc.args)})")
                if event.get_function_responses():
                    fr = event.get_function_responses()[0]
                    logger_orch.info(f"    ADK FR (Orch Test): {fr.name} -> {str(fr.response)[:100]}...")
                if event.is_final_response() and event.content and event.content.parts and event.content.parts[0].text:
                    final_response_text = event.content.parts[0].text.strip()
                    logger_orch.info(f"<<< AURA (Orch Test): {final_response_text}")
                    break
        if final_response_text is None:
            logger_orch.info("<<< AURA (Orch Test): (No textual final response for this turn)")
    logger_orch.info("\n--- Orchestrator Module Test Conversation Ended ---")


if __name__ == "__main__":
    logger_orch.info("Running orchestrator_adk_agent.py directly for testing ADK components.")
    logger_orch.info("This will NOT start a web server.")

    # Basic environment checks relevant for this module's direct testing
    if not os.getenv("OPENROUTER_API_KEY"): logger_orch.error("OPENROUTER_API_KEY (for ADK Agent's LLM) is not set.")
    if APPAGENTX_LEARNING_AVAILABLE or APPAGENTX_DEPLOYMENT_AVAILABLE:
        if not os.getenv("APPAGENTX_LLM_API_KEY") or os.getenv("APPAGENTX_LLM_API_KEY") == "sk-":
            logger_orch.warning("APPAGENTX_LLM_API_KEY not set or default; AppAgentX tools might not function fully.")

    asyncio.run(run_orchestrator_module_test_conversation())