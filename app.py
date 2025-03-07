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

# ✅ Section: Upload Knowledge Base File
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.title("🤖Math assistant")

# ✅ Initialize Chat History in Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ✅ User Input Section
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get AI Response
    ai_response = query_llama3(user_input)

    # Append AI message to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(ai_response)
