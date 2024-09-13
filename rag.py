import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the language model
@st.cache_resource
def init_llm():
    return ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768")

llm = init_llm()
prompt = hub.pull("hwchase17/openai-functions-agent")

# Load and split documents
@st.cache_resource
def init_vectordb():
    loader = WebBaseLoader("https://dev.mrdbourke.com/tensorflow-deep-learning/")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_documents(documents, embeddings)

vectordb = init_vectordb()
retriever = vectordb.as_retriever()

# Create a retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "tensorflow_search",
    "Search for information about TensorFlow. For any questions about TensorFlow, you must use this tool!"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with tools, prompt, and memory
tools = [retriever_tool]
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Streamlit UI
st.title("Interactive ChatGroq with LangChain Demo")

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about TensorFlow"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = agent_executor.invoke({"input": prompt})
        full_response = response["output"]
        message_placeholder.markdown(full_response)
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.experimental_rerun()

# Add a note about session management
st.sidebar.markdown("""
## Session Management
- Your chat history is stored for the duration of this session.
- The session will be cleared if you refresh the page or click 'Clear Chat'.
- Each new question takes into account the context of previous interactions.
""")