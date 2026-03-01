"""
Conversation Memory Manager
Handles short-term (session) and long-term (persistent) memory
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import config


class ConversationMemory:
    def __init__(self):
        """Initialize memory system"""
        self.memory_file = config.MEMORY_FILE
        self.current_session = []
        self.long_term_facts = self._load_long_term_memory()
    
    def _load_long_term_memory(self) -> Dict:
        """Load persistent memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Could not load memory file. Starting fresh.")
                return self._initialize_memory_structure()
        else:
            return self._initialize_memory_structure()
    
    def _initialize_memory_structure(self) -> Dict:
        """Create initial memory structure"""
        return {
            "user_facts": {},  # Key facts about the user
            "conversation_summaries": [],  # Past conversation summaries
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_long_term_memory(self):
        """Save memory to disk"""
        self.long_term_facts["last_updated"] = datetime.now().isoformat()
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.long_term_facts, f, indent=2, ensure_ascii=False)
    
    def add_message(self, user_msg: str, assistant_msg: str):
        """Add a message pair to current session"""
        self.current_session.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_conversation_history(self) -> List[tuple]:
        """
        Get conversation history in Gradio format
        
        Returns:
            List of (user_message, assistant_message) tuples
        """
        return [(msg["user"], msg["assistant"]) for msg in self.current_session]
    
    def get_recent_context(self, num_messages: int = None) -> str:
        """
        Get recent conversation as text context
        
        Args:
            num_messages: Number of recent message pairs to include
        
        Returns:
            Formatted conversation context
        """
        num = num_messages or config.MAX_CONVERSATION_HISTORY
        recent = self.current_session[-num:] if len(self.current_session) > num else self.current_session
        
        context = []
        for msg in recent:
            context.append(f"User: {msg['user']}")
            context.append(f"Assistant: {msg['assistant']}")
        
        return "\n".join(context)
    
    def add_user_fact(self, category: str, key: str, value: str):
        """
        Store a fact about the user
        
        Args:
            category: Category like 'interests', 'beliefs', 'preferences'
            key: Specific fact key
            value: Fact value
        """
        if category not in self.long_term_facts["user_facts"]:
            self.long_term_facts["user_facts"][category] = {}
        
        self.long_term_facts["user_facts"][category][key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
        self.save_long_term_memory()
    
    def get_user_facts(self, category: Optional[str] = None) -> Dict:
        """
        Retrieve user facts
        
        Args:
            category: Optional category filter
        
        Returns:
            User facts dictionary
        """
        if category:
            return self.long_term_facts["user_facts"].get(category, {})
        return self.long_term_facts["user_facts"]
    
    def add_conversation_summary(self, summary: str):
        """
        Add a summary of past conversation
        
        Args:
            summary: Text summary of conversation
        """
        self.long_term_facts["conversation_summaries"].append({
            "summary": summary,
            "created_at": datetime.now().isoformat()
        })
        self.save_long_term_memory()
    
    def get_memory_context(self) -> str:
        """
        Get relevant long-term memory as context for the LLM
        
        Returns:
            Formatted memory context
        """
        context_parts = []
        
        # Add user facts
        if self.long_term_facts["user_facts"]:
            context_parts.append("=== What I know about you ===")
            for category, facts in self.long_term_facts["user_facts"].items():
                context_parts.append(f"\n{category.title()}:")
                for key, data in facts.items():
                    context_parts.append(f"  - {key}: {data['value']}")
        
        # Add recent conversation summaries
        if self.long_term_facts["conversation_summaries"]:
            recent_summaries = self.long_term_facts["conversation_summaries"][-3:]
            context_parts.append("\n=== Recent conversation summaries ===")
            for summary in recent_summaries:
                context_parts.append(f"- {summary['summary']}")
        
        return "\n".join(context_parts) if context_parts else "No previous memory."
    
    def clear_session(self):
        """Clear current session (for new conversation)"""
        self.current_session = []
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "current_session_messages": len(self.current_session),
            "total_user_facts": sum(len(facts) for facts in self.long_term_facts["user_facts"].values()),
            "conversation_summaries": len(self.long_term_facts["conversation_summaries"]),
            "memory_file_exists": self.memory_file.exists()
        }


if __name__ == "__main__":
    # Test memory system
    memory = ConversationMemory()
    print("Memory stats:", memory.get_stats())
    
    # Test adding a fact
    memory.add_user_fact("interests", "programming", "Loves Python and AI")
    print("\nUser facts:", memory.get_user_facts())
    
    # Test memory context
    print("\nMemory context:")
    print(memory.get_memory_context())
