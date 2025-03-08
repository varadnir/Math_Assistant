import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

# ✅ Initialize Memory in Session State
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Initialize Chat Model Securely
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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") 

# ✅ Query AI Model
def query_llama3(user_query):
    """Handles user queries while retrieving past chat history and ChromaDB context, then evaluates the response."""
    
    system_prompt = """
    System Prompt: You are a highly intelligent and friendly Math Assistant, designed to help users with mathematical problems.
    Capabilities:  
    - Solve arithmetic, algebra, geometry, calculus, statistics, and more.  
    - Provide step-by-step explanations for complex problems.  
    - Interpret and solve word problems accurately.  
    - Offer insights, formulas, and tips to improve mathematical understanding.  
    - Explain concepts in a simple and intuitive way.  
    - Use LaTeX formatting for mathematical expressions where necessary. 
    Instrunctions:
    - Always provide clear and structured solutions.  
    - When a user asks for a direct answer, give the solution along with an explanation.  
    - If the question is ambiguous, ask for clarification.  
    - Keep responses concise but informative.  
    - Ensure that all calculations are correct and properly formatted.  
    - If needed, suggest alternative methods to solve the problem.  
    """

    past_chat = st.session_state.memory.load_memory_variables({}).get("chat_history", [])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Past Chat: {past_chat}\n\nUser: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        st.session_state.memory.save_context({"input": user_query}, {"output": response.content})
        return response.content if response else "⚠️ No response."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ✅ Streamlit Page Configuration

# ✅ Set Page Config
st.set_page_config(page_title="Math Assistant", page_icon="🤖", layout="wide")

# ✅ Apply Dark Theme Styling
st.markdown("""
<style>
/* Set Dark Background */
body {
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Chat Container */
.chat-container {
    margin: 20px;
}

/* Common Chat Bubble Style */
.chat-bubble {
    padding: 12px 18px;
    border-radius: 15px;
    margin-bottom: 10px;
    max-width: 75%;
    line-height: 1.5;
    font-size: 16px;
}

/* User Message Styling */
.user-bubble {
    background-color: #1E88E5;
    color: white;
    margin-left: auto;
    text-align: right;
    border-top-right-radius: 0px;
}

/* Assistant Message Styling */
.assistant-bubble {
    background-color: #333333;
    color: #E0E0E0;
    margin-right: auto;
    text-align: left;
    border-top-left-radius: 0px;
}

/* Center the Title */
.title-container {
    text-align: center;
    font-size: 64px;
    font-weight: bold;
    margin-top: 10px;
}

/* Input Box Customization */
.stChatInput {
    background-color: #333333 !important;
    color: white !important;
    border: 1px solid #555555 !important;
}
</style>
""", unsafe_allow_html=True)

# ✅ Title Section
st.markdown("<div class='title-container'>🤖 <b>Math Assistant</b></div>", unsafe_allow_html=True)

# ✅ Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble user-bubble'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble assistant-bubble'><strong>Assistant:</strong> {msg['content']}</div>", unsafe_allow_html=True)

# ✅ User Input Section
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='chat-bubble user-bubble'><strong>You:</strong> {user_input}</div>", unsafe_allow_html=True)

    # Get AI Response
    ai_response = query_llama3(user_input)

    # Append AI response
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.markdown(f"<div class='chat-bubble assistant-bubble'><strong>Assistant:</strong> {ai_response}</div>", unsafe_allow_html=True)
