"""
Conversation Summarizer - Uses Falcon-3 to create concise summaries
Simple and focused
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import config


class ConversationSummarizer:
    def __init__(self):
        """Initialize Falcon-3 for summarization"""
        self.llm = ChatOllama(
            model=config.SUMMARY_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.5
        )
    
    def summarize(self, conversation_text: str) -> str:
        """
        Summarize a conversation concisely
        
        Args:
            conversation_text: The conversation to summarize
        
        Returns:
            Brief summary (1-2 sentences)
        """
        prompt = f"""Summarize this conversation in 1-2 sentences. Focus on the main topics discussed.

Conversation:
{conversation_text}

Summary:"""

        messages = [
            SystemMessage(content="You create brief, accurate summaries."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"Summarization error: {e}")
            return "Conversation about various topics."


# Quick test
if __name__ == "__main__":
    summarizer = ConversationSummarizer()
    
    convo = """User: Hi! I'm learning Python and AI.
Assistant: That's exciting! What aspects interest you most?
User: I love machine learning and want to build my own models.
Assistant: Great! Have you tried any ML libraries yet?
User: Yes, I've been using scikit-learn and TensorFlow."""
    
    print("Summarizing conversation...")
    summary = summarizer.summarize(convo)
    print(f"Summary: {summary}")
