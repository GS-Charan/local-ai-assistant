"""
Local AI Assistant - Phase 2
Auto-summarization and smart memory extraction
"""

import gradio as gr
from llm_handler import OllamaHandler
from conversation_memory import ConversationMemory
from fact_extractor import FactExtractor
from conversation_summarizer import ConversationSummarizer
import config


# Initialize components
llm_handler = OllamaHandler()
memory = ConversationMemory()
fact_extractor = FactExtractor()
summarizer = ConversationSummarizer()


def chat_response(message: str, history: list):
    """
    Handle chat messages with streaming + auto fact extraction + RAG retrieval
    
    Args:
        message: User's message
        history: Gradio chat history
    
    Yields:
        Streaming response chunks
    """
    try:
        # PHASE 2: Retrieve relevant facts from ChromaDB before responding
        relevant_context = memory.get_relevant_context(message)
        
        # If we have relevant memories, prepend them to history
        enhanced_history = history.copy() if history else []
        if relevant_context:
            # Add context as a system-like message at the start
            context_message = f"[Relevant memories about user: {relevant_context}]"
            # Insert at beginning so AI sees it
            enhanced_history.insert(0, (context_message, "Understood, I'll keep that in mind."))
        
        # Stream the response with enhanced context
        final_response = ""
        for partial_response in llm_handler.chat_stream(message, enhanced_history):
            final_response = partial_response
            yield partial_response
        
        # After streaming is complete, save to memory
        if final_response:
            memory.add_message(message, final_response)
            
            # Phase 2: Auto-extract facts every 3 messages
            if len(memory.current_session) % 3 == 0:
                extract_and_store_facts()
            
            # Phase 2: Auto-summarize every 10 messages
            if len(memory.current_session) % 10 == 0:
                auto_summarize()
                
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nPlease try again or check if Ollama is running."
        yield error_msg


def extract_and_store_facts():
    """Extract facts from recent conversation and store them"""
    # Get last 6 messages (3 exchanges)
    recent = memory.current_session[-6:] if len(memory.current_session) >= 6 else memory.current_session
    
    # Format for extraction
    convo_text = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in recent])
    
    # Extract facts
    facts = fact_extractor.extract_facts(convo_text)
    
    # Get existing facts to check contradictions
    existing_facts = [f['fact'] for f in memory.get_all_facts()]
    
    # Store each fact (with contradiction check)
    for fact in facts:
        check = fact_extractor.check_contradiction(fact, existing_facts)
        
        if check['action'] == 'add':
            memory.add_user_fact(fact, category="auto_extracted")
            print(f"✓ Stored: {fact}")
            
        elif check['action'] == 'update':
            print(f"⚠ Contradiction: '{fact}' conflicts with '{check['conflicting_fact']}'")
            
            # Delete the old conflicting fact
            deleted = memory.delete_fact_by_content(check['conflicting_fact'])
            if deleted:
                print(f"  → Deleted old fact")
            
            # Store the new one
            memory.add_user_fact(fact, category="auto_extracted")
            print(f"✓ Stored updated fact: {fact}")
        else:
            # Ignore
            print(f"⊝ Skipped: {fact} (not meaningful)")



def auto_summarize():
    """Auto-summarize conversation when it gets long"""
    # Get all session messages
    convo_text = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" 
                           for m in memory.current_session])
    
    # Create summary
    summary = summarizer.summarize(convo_text)
    
    # Store summary in ChromaDB
    memory.add_conversation_summary(summary)
    print(f"📝 Summary created: {summary}")


def clear_conversation():
    """Clear current conversation session"""
    memory.clear_session()
    return None  # Clears Gradio chat


def get_memory_info():
    """Get memory statistics and context"""
    stats = memory.get_stats()
    context = memory.get_memory_context()
    
    info = f"""📊 **Memory Statistics:**
- Messages in current session: {stats['current_session_messages']}
- Total facts stored: {stats['total_user_facts']}
- Conversation summaries: {stats['conversation_summaries']}
- ChromaDB location: {stats['chromadb_location']}

📝 **Long-term Memory:**
{context if context != "No previous memory." else "No facts stored yet. Have a conversation and I'll remember things about you!"}
"""
    return info


def check_ollama_status():
    """Check if Ollama is connected"""
    success, message = llm_handler.check_connection()
    status = "✅" if success else "❌"
    return f"{status} {message}"


# Create Gradio interface
with gr.Blocks(title="Local AI Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🤖 Local AI Assistant - Phase 2
    
    Your personal AI with auto-summarization and smart memory extraction.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.ChatInterface(
                fn=chat_response,
                type="messages",
                title="💬 Conversation",
                description="Chat with your local AI assistant. It will remember facts about you across sessions!",
                examples=[
                    "Hi! I'm learning Python and AI. What can you help me with?",
                    "I love sci-fi movies, especially ones about AI and space exploration.",
                    "What do you remember about me?",
                ],
                clear_btn="🗑️ Clear Current Session",
                retry_btn="🔄 Retry",
                undo_btn="↩️ Undo",
                concurrency_limit=1,  # Process one message at a time
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Memory & Status")
            
            # Status check
            status_output = gr.Textbox(
                label="Ollama Status",
                lines=3,
                interactive=False
            )
            check_status_btn = gr.Button("🔍 Check Status", size="sm")
            check_status_btn.click(
                fn=check_ollama_status,
                outputs=status_output
            )
            
            gr.Markdown("---")
            
            # Memory info
            memory_output = gr.Markdown(
                label="Memory Info",
                value=get_memory_info()
            )
            refresh_memory_btn = gr.Button("🔄 Refresh Memory", size="sm")
            refresh_memory_btn.click(
                fn=get_memory_info,
                outputs=memory_output
            )
            
            gr.Markdown("---")
            
            # Info and tips
            gr.Markdown("""
            **💡 Tips:**
            - Tell me about your interests
            - Share your beliefs and preferences
            - I'll remember across sessions
            - Ask "what do you remember about me?"
            
            **⚙️ Current Model:**
            `{}`
            """.format(config.CONVERSATION_MODEL))
    
    # Check status on load
    app.load(fn=check_ollama_status, outputs=status_output)


if __name__ == "__main__":
    print("🚀 Starting Local AI Assistant...")
    print(f"📍 Running on http://localhost:{config.GRADIO_SERVER_PORT}")
    print(f"🤖 Model: {config.CONVERSATION_MODEL}")
    print(f"💾 ChromaDB: {config.CHROMA_DB_DIR}")
    print("\n" + "="*50)
    
    # Launch the app
    app.launch(
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        inbrowser=True  # Auto-open browser
    )