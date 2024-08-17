import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import uuid
import random

# Load environment variables
load_dotenv()

# Retrieve Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Available models
MODELS = [
    "gemma-7b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "whisper-large-v3",
]

# Custom CSS for light, minimalistic theme
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 18px !important;
    }
    .small-font {
        font-size: 14px !important;
    }
    .prompt-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .prompt-button {
        background-color: #f0f0f0;
        color: #333333;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .prompt-button:hover {
        background-color: #e0e0e0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton>button {
        border-radius: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .stSelectbox>div>div>div {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "model" not in st.session_state:
        st.session_state.model = MODELS[0]
    if "is_new_session" not in st.session_state:
        st.session_state.is_new_session = True

def create_conversation(model):
    """Create a new conversation chain with the specified model and unlimited memory."""
    memory = ConversationBufferMemory()
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model)
    return ConversationChain(llm=groq_chat, memory=memory)

def get_bot_response(user_input):
    """Get a response from the bot using the current conversation."""
    response = st.session_state.conversation(user_input)
    return response["response"]

def generate_title(chat_id):
    """Generate a title for the chat using AI."""
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chats[chat_id]["messages"]])
    prompt = f"Based on the following conversation, generate a short, concise title (max 6 words):\n\n{conversation_text}\n\nTitle:"
    response = st.session_state.conversation(prompt)
    return response["response"].strip()

def add_chat_to_list(chat_id, messages, title):
    """Add a new chat to the list of chats."""
    st.session_state.chats[chat_id] = {"messages": messages, "title": title}

def delete_chat(chat_id):
    """Delete the specified chat."""
    del st.session_state.chats[chat_id]
    if st.session_state.current_chat_id == chat_id:
        if st.session_state.chats:
            st.session_state.current_chat_id = next(iter(st.session_state.chats))
        else:
            st.session_state.current_chat_id = None
            st.session_state.is_new_session = True

def generate_prompts():
    """Generate a list of example prompts."""
    prompts = [
        "Explain quantum computing in simple terms.",
        "What are the main causes of climate change?",
        "How does machine learning work?",
        "What are the benefits of meditation?",
        "Describe the process of photosynthesis.",
        "What are the key events of World War II?",
        "How does the human immune system function?",
        "Explain the theory of relativity.",
        "What are the major art movements in history?",
        "How do cryptocurrencies work?"
    ]
    return random.sample(prompts, 4)  # Return 4 random prompts

def main():
    st.title("🤖 LLM-powered Chatbot with Groq")

    initialize_session_state()

    # Sidebar for chat management and settings
    with st.sidebar:
        st.title("⚙️ Chatbot Settings")
        model = st.selectbox("🚀 Choose a model", MODELS, index=MODELS.index(st.session_state.model))
        if st.button("➕ New Chat"):
            st.session_state.is_new_session = True
            st.session_state.current_chat_id = None
            st.rerun()

        st.title("💬 Chats")
        for chat_id, chat_data in st.session_state.chats.items():
            col1, col2 = st.columns([3, 1])
            if col1.button(f"📄 {chat_data['title']}", key=f"chat_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.session_state.is_new_session = False
                st.rerun()
            if col2.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

    if model != st.session_state.model:
        st.session_state.model = model
        st.session_state.conversation = create_conversation(model)

    # Main chat area
    if st.session_state.is_new_session:
        st.markdown('<p class="medium-font">🔎 Try these examples:</p>', unsafe_allow_html=True)
        st.markdown('<div class="prompt-container">', unsafe_allow_html=True)
        prompts = generate_prompts()
        for prompt in prompts:
            if st.button(prompt, key=prompt, help=prompt):
                st.session_state.is_new_session = False
                st.session_state.current_chat_id = str(uuid.uuid4())
                st.session_state.conversation = create_conversation(st.session_state.model)
                add_chat_to_list(st.session_state.current_chat_id, [], "New Chat")
                st.session_state.chats[st.session_state.current_chat_id]["messages"].append({"role": "user", "content": prompt})
                bot_response = get_bot_response(prompt)
                st.session_state.chats[st.session_state.current_chat_id]["messages"].append({"role": "assistant", "content": bot_response})
                title = generate_title(st.session_state.current_chat_id)
                st.session_state.chats[st.session_state.current_chat_id]["title"] = title
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.current_chat_id:
        for message in st.session_state.chats[st.session_state.current_chat_id]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input
    user_input = st.chat_input("💬 Type your message here...")
    if user_input:
        if st.session_state.is_new_session:
            st.session_state.is_new_session = False
            st.session_state.current_chat_id = str(uuid.uuid4())
            st.session_state.conversation = create_conversation(st.session_state.model)
            add_chat_to_list(st.session_state.current_chat_id, [], "New Chat")

        st.session_state.chats[st.session_state.current_chat_id]["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            bot_response = get_bot_response(user_input)
            st.markdown(bot_response)

        st.session_state.chats[st.session_state.current_chat_id]["messages"].append({"role": "assistant", "content": bot_response})

        if len(st.session_state.chats[st.session_state.current_chat_id]["messages"]) == 2:
            title = generate_title(st.session_state.current_chat_id)
            st.session_state.chats[st.session_state.current_chat_id]["title"] = title

        st.rerun()

if __name__ == "__main__":
    main()