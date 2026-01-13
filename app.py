from config.load_env import load_env
from rag_assistant.rag_assistant import RAGAssistant
from rag_assistant.gradio_interface import GradioInterface
from gradio import themes
import gradio as gr
from gradio.themes.utils import sizes
from utils.load_yaml_config import load_all_prompts
from config.paths import PROMPTS_DIR

load_env()

# Load components from components.yaml for reusable prompt elements
try:
    all_prompts = load_all_prompts(PROMPTS_DIR)
    # Extract components (tools, tones, reasoning_strategies)
    components = {
        'tools': all_prompts.get('tools'),
        'tones': all_prompts.get('tones'),
        'reasoning_strategies': all_prompts.get('reasoning_strategies')
    }
except Exception as e:
    print(f"Warning: Could not load components: {e}")
    components = None  # Will use defaults

# Initialize RAG Assistant with new modular structure
# Updated for Blueprint Analytics textbook with educational_assistant prompt
assistant = RAGAssistant(
    persist_path="./chroma/rag",
    collection_name="blueprint_text_analytics",
    topic="Blueprint Text Analytics in Python textbook",
    prompt_template='educational_assistant',
    components=components  # Includes tools, tones, reasoning_strategies
)

gradio_assistant = GradioInterface(assistant)
logger = gradio_assistant.logger

def gradio_chat(message, history):
    """Wrapper with loading state."""
    try:
        # Show typing indicator
        yield history + [{"role": "assistant", "content": "⏳ Thinking..."}]
        
        # Start streaming actual response
        accumulated = ""
        for chunk in gradio_assistant.stream_chat(message, history, 3):
            accumulated = chunk
            yield history + [{"role": "assistant", "content": accumulated}]
    except Exception as e:
        logger.exception("Error in chat")
        yield history + [{"role": "assistant", "content": f"❌ Error: {str(e)}"}]

# Customize the chatbot display
custom_chatbot = gr.Chatbot(
    height=600,                    # Height in pixels
    show_label=False,              # Hide the "Chatbot" label above
    avatar_images=("assets/blueprint.png", "assets/chat.png"),
    editable='user',       # Keep bubbles from stretching full width
    buttons=['copy'],         # Add copy button to messages
    watermark='Taylor Kirk',
    layout="bubble",                # "panel" or "bubble" style
    placeholder="Start chatting!"  # Message shown when chat is empty
)

# Customize the input textbox
custom_textbox = gr.Textbox(
    placeholder="Type your message here...",
    container=True,               # Remove the container border
    scale=7,                       # Takes up more horizontal space relative to buttons
    lines=1,                       # Number of visible lines (expands as user types)
    max_lines=10,                  # Maximum lines before scrolling
    autofocus=True,                 # Auto-focus on page load
    submit_btn='Submit',
    stop_btn='Stop'
)

custom_theme = gr.themes.Soft(# type: ignore
    primary_hue="blue",
    secondary_hue="slate",
    font=gr.themes.GoogleFont("Inter"), # type: ignore
)

# Customize CSS
custom_css = """
.message-wrap {
    border-radius: 12px !important;
}
.bot-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}
#component-0 {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Hide footer */
footer {visibility: hidden}

/* Gradient animation */
@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Target the title h1 */
h1 {
    background-image: linear-gradient(270deg, #0066CC, #00A3E0, #667eea, #4facfe) !important;
    background-size: 400% 400% !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;  /* Keep this separate! */
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    animation: gradient-shift 8s ease infinite !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 0.3rem !important;
    letter-spacing: -0.01em !important;
}

/* Target description BUT NOT chat messages */
.md.prose:not(.chatbot) p {
    background-image: linear-gradient(270deg, #8b5cf6, #a855f7, #d946ef, #ec4899) !important;
    background-size: 400% 400% !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;  /* Keep this separate! */
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    animation: gradient-shift 8s ease infinite !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    text-align: center !important;
    margin-top: 0 !important;
}

/* Chat bubbles polish */
.message-wrap {
    border-radius: 12px !important;
}
"""

demo = gr.ChatInterface(
    fn=gradio_chat,
    title="Blueprints for Text Analytics in Python Textbook",
    chatbot=custom_chatbot,
    textbox=custom_textbox,
    editable=True,
    description="Ask questions about NLP solutions for real world problems",
    examples=[
        "How can I build a simple preprocessing pipeline for text data?",
        "What are n-grams and how are they relevant to machine learning?",
        "What are popular libraries in python for NLP?"
    ]
)

if __name__ == "__main__":

    try:
        demo.launch(share=True, server_name="127.0.0.1", server_port=7860, theme=custom_theme, css=custom_css)
    except Exception as e:
        print(f"Failed to launch: {e}")
        logger.exception(f"Failure to launch {e}")