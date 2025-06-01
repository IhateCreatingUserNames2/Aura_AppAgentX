# a2a_wrapper/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime, timezone
import uuid

class A2APart(BaseModel):
    type: str # "text", "file", "data"
    text: Optional[str] = None
    # For file: {"name": "optional", "mimeType": "optional", "bytes": "base64" or "uri": "url"}
    file: Optional[Dict[str, Any]] = None
    # For data: any JSON serializable dict
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class A2AMessage(BaseModel):
    role: str # "user" or "agent"
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None

class A2ATaskSendParams(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())) # Task ID for A2A
    sessionId: Optional[str] = None # Optional session ID for A2A (client managed)
    message: A2AMessage
    historyLength: Optional[int] = None
    # pushNotification: Optional[Dict[str, Any]] = None # For brevity
    metadata: Optional[Dict[str, Any]] = None

class A2AArtifact(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None
    index: Optional[int] = 0
    append: Optional[bool] = False
    lastChunk: Optional[bool] = False

class A2ATaskStatus(BaseModel):
    state: str # e.g., "completed", "failed", "working", "input-required"
    message: Optional[A2AMessage] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class A2ATaskResult(BaseModel):
    id: str # Task ID (should match the one in A2ATaskSendParams)
    sessionId: Optional[str] = None # Echoed from A2ATaskSendParams
    status: A2ATaskStatus
    artifacts: Optional[List[A2AArtifact]] = None
    history: Optional[List[A2AMessage]] = None # Optional history of messages in this task
    metadata: Optional[Dict[str, Any]] = None

class A2AJsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int] # Request ID for JSON-RPC
    method: str
    params: Optional[A2ATaskSendParams] = None # Specifically for tasks/send

class A2AJsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int] # Must match request ID
    result: Optional[A2ATaskResult] = None
    error: Optional[Dict[str, Any]] = None

# --- Agent Card Models (from A2A Protocol ReadMe) ---
class AgentCardProvider(BaseModel):
    organization: str
    url: str

class AgentCardCapabilities(BaseModel):
    streaming: Optional[bool] = False
    pushNotifications: Optional[bool] = False
    stateTransitionHistory: Optional[bool] = False

class AgentCardAuthentication(BaseModel):
    schemes: List[str] # e.g. ["Bearer", "OAuth2"]
    credentials: Optional[str] = None # "credentials a client should use for private cards"

class AgentCardSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: Optional[List[str]] = []
    examples: Optional[List[str]] = []
    # Parameters for this skill, as a JSON schema object
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    inputModes: Optional[List[str]] = None # Mime types
    outputModes: Optional[List[str]] = None # Mime types


class AgentCard(BaseModel):
    name: str
    description: str
    url: str # URL where the agent is hosted (this A2A wrapper server's base URL)
    provider: Optional[AgentCardProvider] = None
    version: str = "1.0.0"
    documentationUrl: Optional[str] = None
    capabilities: AgentCardCapabilities = Field(default_factory=AgentCardCapabilities)
    authentication: AgentCardAuthentication = Field(default_factory=lambda: AgentCardAuthentication(schemes=[]))
    defaultInputModes: List[str] = ["application/json"] # A2A server expects JSON-RPC
    defaultOutputModes: List[str] = ["application/json"] # A2A server responds with JSON-RPC
    skills: List[AgentCardSkill]