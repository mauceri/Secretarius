from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Awaitable
from .models import Message

class LLMInterface(ABC):
    @abstractmethod
    async def generate_response(self, messages: List[Message], system_prompt: str) -> str:
        """Send messages to the LLM and get the raw string response."""
        pass

class ToolClientInterface(ABC):
    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools with their schemas."""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific tool by name."""
        pass

    @abstractmethod
    async def connect(self):
        """Connect to the tools server."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the tools server."""
        pass

class InputGatewayInterface(ABC):
    @abstractmethod
    async def display_thought(self, thought: str):
        """Log or display the intermediate reasoning steps."""
        pass

    @abstractmethod
    async def display_message(self, role: str, content: str):
        """Display a final or conversational message to the user."""
        pass

    @abstractmethod
    def set_callback(self, callback: Callable[[str], Awaitable[None]]):
        """Set the orchestrator callback to handle user inputs."""
        pass

    @abstractmethod
    async def run(self):
        """Run the gateway interface loop."""
        pass
