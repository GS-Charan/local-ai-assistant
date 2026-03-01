"""
LLM Handler for Ollama integration
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Generator
import config


class OllamaHandler:
    def __init__(self, model_name: str = None):
        """Initialize Ollama chat model"""
        self.model_name = model_name or config.CONVERSATION_MODEL
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            num_predict=config.MAX_TOKENS,
            top_p=config.TOP_P,
        )
    
    def chat(self, message: str, history: List[tuple] = None) -> str:
        """
        Send a message and get response (non-streaming)
        
        Args:
            message: User's message
            history: List of (user_msg, assistant_msg) tuples
        
        Returns:
            Assistant's response
        """
        messages = self._build_messages(message, history)
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}\n\nMake sure Ollama is running and the model is loaded."
    
    def chat_stream(self, message: str, history: List[tuple] = None) -> Generator[str, None, None]:
        """
        Send a message and stream response
        
        Args:
            message: User's message
            history: List of (user_msg, assistant_msg) tuples
        
        Yields:
            Chunks of assistant's response
        """
        messages = self._build_messages(message, history)
        
        try:
            full_response = ""
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    yield full_response
        except Exception as e:
            yield f"Error communicating with Ollama: {str(e)}\n\nMake sure Ollama is running and the model is loaded."
    
    def _build_messages(self, message: str, history: List = None) -> List:
        """
        Build message list for LangChain
        
        Args:
            message: Current user message
            history: Previous conversation history (Gradio 5.x messages format or tuples)
        
        Returns:
            List of message objects
        """
        messages = [SystemMessage(content=config.SYSTEM_PROMPT)]
        
        # Add conversation history
        if history:
            # Take only recent history to avoid context overflow
            recent_history = history[-config.MAX_CONVERSATION_HISTORY:]
            
            for item in recent_history:
                # Handle both Gradio 5.x messages format and legacy tuple format
                if isinstance(item, dict):
                    # Gradio 5.x format: {"role": "user"/"assistant", "content": "..."}
                    if item.get("role") == "user":
                        messages.append(HumanMessage(content=item["content"]))
                    elif item.get("role") == "assistant":
                        messages.append(AIMessage(content=item["content"]))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    # Legacy format: (user_msg, assistant_msg)
                    user_msg, assistant_msg = item
                    if user_msg:
                        messages.append(HumanMessage(content=user_msg))
                    if assistant_msg:
                        messages.append(AIMessage(content=assistant_msg))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        return messages
    
    def check_connection(self) -> tuple[bool, str]:
        """
        Check if Ollama is running and model is available
        
        Returns:
            (success, message)
        """
        import requests
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                return False, "Ollama is not responding"
            
            # Check if model exists
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model_name not in model_names:
                return False, f"Model '{self.model_name}' not found. Available models: {', '.join(model_names)}"
            
            return True, f"Connected to Ollama. Model '{self.model_name}' is ready."
        
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama. Is it running? Start with 'ollama serve'"
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"


# Convenience function for quick testing
def test_connection():
    """Test Ollama connection"""
    handler = OllamaHandler()
    success, message = handler.check_connection()
    print(message)
    return success


if __name__ == "__main__":
    # Test the handler
    test_connection()