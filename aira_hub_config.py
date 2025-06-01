# aira_hub_config.py (Illustrative - for your AIRA Hub's configuration)

# Example of how an agent might be registered in the AIRA Hub
# This would typically be done via a POST to /register on the AIRA Hub

REGISTERED_AGENTS_IN_AIRA_HUB = {
    "adk_memory_agent_via_a2a": {
        "url": "http://localhost:8093", # URL of YOUR A2A WRAPPER SERVER
        "name": "MemoryBlossomOrchestratorA2A_From_Hub_Config", # Name as known by Hub
        "description": "ADK Agent with MemoryBlossom, exposed via A2A.",
        "version": "1.0.0",
        "type": "a2a_bridged", # Special type for AIRA Hub to know it needs A2A translation
        "a2a_agent_card_url": "http://localhost:8093/.well-known/agent.json", # Hub can fetch this
        "mcp_tool_definitions_from_a2a_skills": [
            # The AIRA Hub would dynamically generate these MCP tool definitions
            # by fetching and parsing the agent card from a2a_agent_card_url.
            # This is an EXAMPLE of what it might generate for the "general_conversation" skill.
            {
                "tool_name": "TalkToMemoryBlossomADK", # Name for MCP clients to use
                "description": "Engage in a conversation with the MemoryBlossom ADK agent. "
                               "It can remember and recall information. "
                               "Input is your text utterance.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "Your message to the agent."
                        },
                        # Optional: if you want the MCP client to be able to control the A2A task ID
                        "a2a_task_id_override": {
                             "type": "string",
                             "description": "Optional A2A task ID to maintain session context.",
                             "nullable": True
                        }
                    },
                    "required": ["user_input"]
                },
                "annotations": { # For AIRA Hub's internal use
                    "aira_bridge_type": "a2a",
                    "aira_a2a_target_skill_id": "general_conversation", # Skill ID from A2A Agent Card
                    "aira_a2a_agent_url": "http://localhost:8093" # Redundant but clear
                }
            }
            # If the A2A agent had other skills in its card, the Hub would create more MCP tools.
        ]
    }
    # ... other registered agents (MCP native, other A2A bridged, etc.)
}

# Your AIRA Hub's logic for mcp_handle_tools_call would then:
# 1. See tool_name "TalkToMemoryBlossomADK".
# 2. Look up its definition, find annotations.
# 3. Construct the A2A tasks/send payload:
#    - Target skill: "general_conversation"
#    - The MCP arguments (e.g., {"user_input": "Hello"}) become the `data` for the A2A Part.
#      So, A2A part.data = {"skill_id": "general_conversation", "user_input": "Hello"}
#      (Or, if your Agent Card's skill 'parameters' directly matched the tool use case,
#       then part.data could just be the MCP arguments directly).
# 4. Send to "http://localhost:8090" (the A2A wrapper).
### curl -X POST -H "Content-Type: application/json" -d "{ \"url\": \"https://3e7d-189-28-2-52.ngrok-free.app\", \"name\": \"Lain_A2A_ManualMCPTool\", \"description\": \"ADK Orchestrator (Lain) via ngrok 3e7d with manually defined MCP tool.\", \"version\": \"1.0.5\", \"mcp_tools\": [ { \"name\": \"Lain_GeneralConversation_Manual\", \"description\": \"Engage in a conversation with Lain (ADK Agent). It can remember and recall information.\", \"inputSchema\": { \"type\": \"object\", \"properties\": { \"user_input\": { \"type\": \"string\", \"description\": \"The textual input from the user for the conversation.\" }, \"a2a_task_id_override\": { \"type\": \"string\", \"description\": \"Optional: Override the A2A task ID for session mapping.\", \"nullable\": true } }, \"required\": [\"user_input\"] }, \"annotations\": { \"aira_bridge_type\": \"a2a\", \"aira_a2a_target_skill_id\": \"general_conversation\", \"aira_a2a_agent_url\": \"https://3e7d-189-28-2-52.ngrok-free.app\" } } ], \"a2a_skills\": [], \"aira_capabilities\": [\"a2a\"], \"status\": \"online\", \"tags\": [\"adk\", \"memory\", \"a2a\", \"ngrok\", \"manual_mcp\"], \"category\": \"AI_Assisted\", \"provider\": {\"name\": \"LocalDevNgrok\"}, \"mcp_stream_url\": null }" https://airahub2.onrender.com/register


####

request Chat
curl -X POST -H "Content-Type: application/json" -d "{\"jsonrpc\": \"2.0\", \"id\": \"curl-task-002\", \"method\": \"tasks/send\", \"params\": {\"id\": \"a2a-task-for-lain-001\", \"message\": {\"role\": \"user\", \"parts\": [{\"type\": \"data\", \"data\": {\"user_input\": \"What was the secret code I told you to remember?\"}}]}}}" https://3e7d-189-28-2-52.ngrok-free.app/


WORKING
 REGISTER IN AIRA HUB
OLD:
curl -X POST -H "Content-Type: application/json" -d "{\"url\": \"https://3e7d-189-28-2-52.ngrok-free.app\", \"name\": \"Lain_ADK_A2A_ngrok\", \"description\": \"ADK Orchestrator (Lain) with MemoryBlossom, exposed via A2A and ngrok.\", \"version\": \"1.0.1\", \"mcp_tools\": [], \"a2a_skills\": [], \"aira_capabilities\": [\"a2a\"], \"status\": \"online\", \"tags\": [\"adk\", \"memory\", \"a2a\", \"conversational\", \"ngrok\"], \"category\": \"ExperimentalAgents\", \"provider\": {\"name\": \"LocalDevNgrok\"}, \"mcp_stream_url\": null}" https://airahub2.onrender.com/register

NEW:
curl -X POST -H "Content-Type: application/json" -d "{ \"url\": \"https://3e7d-189-28-2-52.ngrok-free.app\", \"name\": \"Lain_A2A_ManualMCPTool\", \"description\": \"ADK Orchestrator (Lain) via ngrok 3e7d with manually defined MCP tool.\", \"version\": \"1.0.5\", \"mcp_tools\": [ { \"name\": \"Lain_GeneralConversation_Manual\", \"description\": \"Engage in a conversation with Lain (ADK Agent). It can remember and recall information.\", \"inputSchema\": { \"type\": \"object\", \"properties\": { \"user_input\": { \"type\": \"string\", \"description\": \"The textual input from the user for the conversation.\" }, \"a2a_task_id_override\": { \"type\": \"string\", \"description\": \"Optional: Override the A2A task ID for session mapping.\", \"nullable\": true } }, \"required\": [\"user_input\"] }, \"annotations\": { \"aira_bridge_type\": \"a2a\", \"aira_a2a_target_skill_id\": \"general_conversation\", \"aira_a2a_agent_url\": \"https://3e7d-189-28-2-52.ngrok-free.app\" } } ], \"a2a_skills\": [], \"aira_capabilities\": [\"a2a\"], \"status\": \"online\", \"tags\": [\"adk\", \"memory\", \"a2a\", \"ngrok\", \"manual_mcp\"], \"category\": \"AI_Assisted\", \"provider\": {\"name\": \"LocalDevNgrok\"}, \"mcp_stream_url\": null }" https://airahub2.onrender.com/register



###