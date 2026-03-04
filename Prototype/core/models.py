from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None

class ActionRequest(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]

class ReActStep(BaseModel):
    thought: str = ""
    action: Optional[ActionRequest] = None
    observation: Optional[str] = None

class Priority(int, Enum):
    TOOL_RESULT = 1       # Priorité 1
    SESSION_HISTORY = 2   # Priorité 2
    SEMANTIC_MEMORY = 3   # Priorité 3

class ContextItem(BaseModel):
    priority: Priority
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SessionState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    context_items: List[ContextItem] = Field(default_factory=list)
