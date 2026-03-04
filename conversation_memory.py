"""
Conversation Memory Manager - ChromaDB Version
Simple vector-based memory for semantic search and smart fact storage
"""

import chromadb
from datetime import datetime
from typing import List, Dict, Optional
import config


class ConversationMemory:
    def __init__(self):
        """Initialize ChromaDB-based memory system"""
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        
        # Get or create collections
        self.facts_collection = self.client.get_or_create_collection(
            name="user_facts",
            metadata={"description": "Facts about the user"}
        )
        
        self.conversations_collection = self.client.get_or_create_collection(
            name="conversation_history",
            metadata={"description": "Past conversation summaries"}
        )
        
        # Current session (in-memory, not persisted)
        self.current_session = []
    
    def add_message(self, user_msg: str, assistant_msg: str):
        """
        Add a message pair to current session
        
        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
        """
        self.current_session.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_user_fact(self, fact: str, category: str = "general"):
        """
        Store a fact about the user in ChromaDB
        
        Args:
            fact: The fact to store (e.g., "loves Python programming")
            category: Category like 'interests', 'beliefs', 'preferences'
        """
        fact_id = f"{category}_{datetime.now().timestamp()}"
        
        self.facts_collection.add(
            documents=[fact],
            metadatas=[{
                "category": category,
                "created_at": datetime.now().isoformat()
            }],
            ids=[fact_id]
        )
    
    def search_facts(self, query: str, n_results: int = 5) -> List[str]:
        """
        Search for relevant facts using semantic similarity
        
        Args:
            query: Natural language query
            n_results: Number of results to return
        
        Returns:
            List of relevant facts
        """
        if self.facts_collection.count() == 0:
            return []
        
        results = self.facts_collection.query(
            query_texts=[query],
            n_results=min(n_results, self.facts_collection.count())
        )
        
        if results['documents'] and len(results['documents']) > 0:
            return results['documents'][0]
        return []
    
    def get_all_facts(self) -> List[Dict]:
        """
        Get all stored facts
        
        Returns:
            List of facts with metadata
        """
        if self.facts_collection.count() == 0:
            return []
        
        results = self.facts_collection.get()
        
        facts = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                facts.append({
                    "fact": doc,
                    "category": results['metadatas'][i].get('category', 'general'),
                    "created_at": results['metadatas'][i].get('created_at', 'unknown')
                })
        
        return facts
    
    def add_conversation_summary(self, summary: str):
        """
        Add a summary of past conversation
        
        Args:
            summary: Text summary of conversation
        """
        summary_id = f"summary_{datetime.now().timestamp()}"
        
        self.conversations_collection.add(
            documents=[summary],
            metadatas=[{"created_at": datetime.now().isoformat()}],
            ids=[summary_id]
        )
    
    def search_past_conversations(self, query: str, n_results: int = 3) -> List[str]:
        """
        Search past conversation summaries
        
        Args:
            query: What to search for
            n_results: Number of results
        
        Returns:
            List of relevant conversation summaries
        """
        if self.conversations_collection.count() == 0:
            return []
        
        results = self.conversations_collection.query(
            query_texts=[query],
            n_results=min(n_results, self.conversations_collection.count())
        )
        
        if results['documents'] and len(results['documents']) > 0:
            return results['documents'][0]
        return []
    
    def get_conversation_history(self) -> List[tuple]:
        """
        Get current session history in Gradio format
        
        Returns:
            List of (user_message, assistant_message) tuples
        """
        return [(msg["user"], msg["assistant"]) for msg in self.current_session]
    
    def get_memory_context(self) -> str:
        """
        Get formatted memory context for display
        
        Returns:
            Formatted memory string
        """
        context_parts = []
        
        # Get all facts grouped by category
        all_facts = self.get_all_facts()
        
        if all_facts:
            context_parts.append("=== What I know about you ===")
            
            # Group by category
            categories = {}
            for fact_data in all_facts:
                category = fact_data['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(fact_data['fact'])
            
            # Format by category
            for category, facts in categories.items():
                context_parts.append(f"\n**{category.title()}:**")
                for fact in facts:
                    context_parts.append(f"  • {fact}")
        
        # Get recent conversation summaries
        all_summaries = self.conversations_collection.get()
        if all_summaries['documents']:
            # Get last 3 summaries
            recent = all_summaries['documents'][-3:] if len(all_summaries['documents']) > 3 else all_summaries['documents']
            if recent:
                context_parts.append("\n=== Recent Conversations ===")
                for summary in recent:
                    context_parts.append(f"  • {summary}")
        
        return "\n".join(context_parts) if context_parts else "No memories stored yet."
    
    def get_relevant_context(self, current_message: str) -> str:
        """
        Get relevant memories based on current message
        This is what gets sent to the LLM for context
        
        Args:
            current_message: The user's current message
        
        Returns:
            Relevant context string
        """
        relevant_parts = []
        
        # Search for relevant facts
        relevant_facts = self.search_facts(current_message, n_results=5)
        if relevant_facts:
            relevant_parts.append("Relevant facts about user:")
            for fact in relevant_facts:
                relevant_parts.append(f"- {fact}")
        
        # Search for relevant past conversations
        relevant_convos = self.search_past_conversations(current_message, n_results=2)
        if relevant_convos:
            relevant_parts.append("\nRelevant past discussions:")
            for convo in relevant_convos:
                relevant_parts.append(f"- {convo}")
        
        return "\n".join(relevant_parts) if relevant_parts else ""
    
    def clear_session(self):
        """Clear current session (keeps long-term memory)"""
        self.current_session = []
    
    def clear_all_memory(self):
        """⚠️ DANGER: Delete all stored memories (cannot be undone!)"""
        self.client.delete_collection("user_facts")
        self.client.delete_collection("conversation_history")
        
        # Recreate empty collections
        self.facts_collection = self.client.get_or_create_collection(name="user_facts")
        self.conversations_collection = self.client.get_or_create_collection(name="conversation_history")
        self.current_session = []
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "current_session_messages": len(self.current_session),
            "total_user_facts": self.facts_collection.count(),
            "conversation_summaries": self.conversations_collection.count(),
            "chromadb_location": str(config.CHROMA_DB_DIR)
        }


# Simple usage example
if __name__ == "__main__":
    # Test the memory system
    memory = ConversationMemory()
    
    print("Testing ChromaDB Memory System\n")
    
    # Add some facts
    print("Adding facts...")
    memory.add_user_fact("Loves Python programming", category="interests")
    memory.add_user_fact("Prefers dark mode", category="preferences")
    memory.add_user_fact("Learning AI and machine learning", category="interests")
    
    # Search for facts
    print("\nSearching for 'programming':")
    results = memory.search_facts("programming")
    for fact in results:
        print(f"  - {fact}")
    
    # Get stats
    print("\nMemory stats:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show all facts
    print("\nAll stored facts:")
    all_facts = memory.get_all_facts()
    for fact_data in all_facts:
        print(f"  [{fact_data['category']}] {fact_data['fact']}")