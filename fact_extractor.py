"""
Fact Extractor - Uses Falcon-3 to extract user facts from conversations
Keeps it simple: just extract and store
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import config


class FactExtractor:
    def __init__(self):
        """Initialize Falcon-3 for fact extraction"""
        self.llm = ChatOllama(
            model=config.SUMMARY_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.3  # Lower temp for factual extraction
        )
    
    def extract_facts(self, conversation_text: str) -> list[str]:
        """
        Extract facts about the user from conversation
        
        Args:
            conversation_text: The conversation to analyze
        
        Returns:
            List of facts (e.g., ["loves Python", "prefers dark mode"])
        """
        prompt = f"""Extract ONLY meaningful, non-obvious facts about the user from this conversation.

RULES:
- Skip obvious statements (e.g., "User's name is [Name]" - we already know their name)
- Focus on: interests, preferences, beliefs, habits, skills, dislikes
- Ignore: greetings, questions, temporary states, self-evident facts
- Be concise: "loves Python" not "User loves Python programming"
- One fact per line

Conversation:
{conversation_text}

Meaningful facts (one per line):"""

        messages = [SystemMessage(content="You extract only meaningful facts concisely."), HumanMessage(content=prompt)]
        
        try:
            response = self.llm.invoke(messages)
            # Split by newlines and clean
            facts = [f.strip() for f in response.content.split('\n') if f.strip() and not f.startswith('#') and not f.startswith('-')]
            # Filter out very short or meaningless facts
            facts = [f for f in facts if len(f) > 10]
            return facts[:5]  # Max 5 facts per extraction
        except Exception as e:
            print(f"Fact extraction error: {e}")
            return []
    
    def check_contradiction(self, new_fact: str, existing_facts: list[str]) -> dict:
        """
        Check if new fact contradicts existing ones
        
        Args:
            new_fact: New fact to check
            existing_facts: List of existing facts
        
        Returns:
            {"contradicts": bool, "conflicting_fact": str or None, "action": "update"|"ignore"|"add"}
        """
        if not existing_facts:
            return {"contradicts": False, "conflicting_fact": None, "action": "add"}
        
        prompt = f"""Does this NEW statement contradict any EXISTING facts?

NEW: {new_fact}

EXISTING:
{chr(10).join(f"- {f}" for f in existing_facts)}

Reply ONLY with:
- "NO" if no contradiction
- "YES: <conflicting fact>" if there's a contradiction

Answer:"""

        messages = [SystemMessage(content="You detect contradictions precisely."), HumanMessage(content=prompt)]
        
        try:
            response = self.llm.invoke(messages).content.strip()
            
            if response.upper().startswith("YES"):
                conflicting = response.split(":", 1)[1].strip() if ":" in response else existing_facts[0]
                return {"contradicts": True, "conflicting_fact": conflicting, "action": "update"}
            else:
                return {"contradicts": False, "conflicting_fact": None, "action": "add"}
        except Exception as e:
            print(f"Contradiction check error: {e}")
            return {"contradicts": False, "conflicting_fact": None, "action": "add"}


# Quick test
if __name__ == "__main__":
    extractor = FactExtractor()
    
    # Test extraction
    convo = """User: I love Python programming!
Assistant: That's great! What do you like about it?
User: I prefer it for AI and machine learning projects."""
    
    print("Extracting facts...")
    facts = extractor.extract_facts(convo)
    print("Facts found:")
    for f in facts:
        print(f"  - {f}")
    
    # Test contradiction
    print("\nTesting contradiction...")
    new_fact = "prefers JavaScript over Python"
    existing = ["loves Python programming"]
    result = extractor.check_contradiction(new_fact, existing)
    print(f"Result: {result}")