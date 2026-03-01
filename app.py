"""
Local AI Assistant - Phase 1
Main Gradio application with conversation and memory
"""

import gradio as gr
from llm_handler import OllamaHandler
from conversation_memory import ConversationMemory
import config


# Initialize components
llm_handler = OllamaHandler()
memory = ConversationMemory()


def chat_response(message: str, history: list):
    """
    Handle chat messages with streaming
    
    Args:
        message: User's message
        history: Gradio chat history [(user_msg, assistant_msg), ...]
    
    Yields:
        Streaming response chunks
    """
    # Stream the response
    for partial_response in llm_handler.chat_stream(message, history):
        yield partial_response
    
    # After streaming is complete, save to memory
    final_response = partial_response
    memory.add_message(message, final_response)


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
- Memory file: {'✓ Exists' if stats['memory_file_exists'] else '✗ Not found'}

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
    # 🤖 Local AI Assistant - Phase 1
    
    Your personal AI running locally with persistent memory.
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
    print(f"💾 Memory file: {config.MEMORY_FILE}")
    print("\n" + "="*50)
    
    # Launch the app
    app.launch(
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        inbrowser=True  # Auto-open browser
    )
