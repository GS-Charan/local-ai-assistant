"""
Configuration file for Local AI Assistant
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "memory_store.json"

# Ensure directories exist
MEMORY_DIR.mkdir(exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "dolphin-phi3-medium:latest"  # Dolphin Phi-3 Medium - uncensored & conversational

# Model Configuration
CONVERSATION_MODEL = "dolphin-phi3-medium:latest"  # Main chat model
SUMMARY_MODEL = "falcon3-7b"  # For summarization (Phase 2)
VISION_MODEL = "joycaption"  # For image analysis (Phase 3)

# Memory Configuration
MAX_CONVERSATION_HISTORY = 20  # Number of messages to keep in context
CONVERSATION_SUMMARY_THRESHOLD = 30  # Summarize after this many messages

# Generation Parameters
TEMPERATURE = 0.7
MAX_TOKENS = 2048
TOP_P = 0.9

# Gradio Configuration
GRADIO_SHARE = False  # Set True to create public link
GRADIO_SERVER_PORT = 7860

# System Prompt
SYSTEM_PROMPT = """You are a helpful AI assistant living on the user's local PC. You have persistent memory of past conversations and can learn about the user's beliefs, interests, and preferences over time.

Key behaviors:
- Be conversational, friendly, and natural
- Remember facts the user shares about themselves
- If the user contradicts previous information, politely ask for clarification
- Be thoughtful about whether statements are jokes, sarcasm, or genuine beliefs
- Call out inconsistencies respectfully rather than blindly accepting everything

You are currently in a conversation. Be helpful, engaging, and remember context."""