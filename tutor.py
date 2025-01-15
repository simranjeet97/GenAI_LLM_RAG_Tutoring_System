import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from torch import float16
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import torch

# Ensure the code does not use MPS
torch.device("cpu")

# Constants
TEXTS_DIRECTORY = "text_files"

# Initialize embeddings and Chroma vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db"  # Directory for storing the database
)

# Load and chunk documents
def load_and_chunk_documents_to_vector_store(directory_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            loader = TextLoader(file_path)
            documents = loader.load()
            chunked_documents = text_splitter.split_documents(documents)
            vector_store.add_documents(chunked_documents)

load_and_chunk_documents_to_vector_store(TEXTS_DIRECTORY)

# Dictionary of model names
model_dict = {
    "KingNish/Qwen2.5-0.5b-Test-ft": "KingNish/Qwen2.5-0.5b-Test-ft",
    "microsoft/phi-4": "microsoft/phi-4",
    "mistralai/Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407"
}
# Show Previous Topics Explored
if "all_topics" not in st.session_state:
    st.session_state.all_topics = []  # Initialize with an empty list

if "current_topic" not in st.session_state:
    st.session_state.current_topic = "General"  # Default topic
    st.session_state.topic_changed = False

# Sidebar for topic input
new_topic = st.sidebar.text_input("Enter topic:", st.session_state.current_topic)

# Detect topic change
if new_topic and new_topic != st.session_state.current_topic:
    st.session_state.current_topic = new_topic
    st.session_state.topic_changed = True  # Mark topic as changed
    if new_topic not in st.session_state.all_topics:
        st.session_state.all_topics.append(new_topic)
else:
    st.session_state.topic_changed = False  # Reset flag if topic hasn't changed

# Display tracked topics
st.sidebar.markdown("### Topics Explored:")
for topic in st.session_state.all_topics:
    st.sidebar.write(f"- {topic}")

# Model selection
selected_model_name = st.sidebar.selectbox(
    "Select a Model",
    options=list(model_dict.keys()),
    index=0  # Default to the first model
)

# Maximum tokens for model generation
max_tokens = st.sidebar.slider(
    "Maximum Tokens for Response", min_value=50, max_value=500, value=100, step=50
)

# Temperature for response generation (higher is more random)
temperature = st.sidebar.slider(
    "Temperature (creativity)", min_value=0.0, max_value=1.0, value=0.7, step=0.05
)

# Top-P (nucleus sampling)
top_p = st.sidebar.slider(
    "Top-P (nucleus sampling)", min_value=0.0, max_value=1.0, value=1.0, step=0.05
)

# Language Selection (if the model supports multiple languages)
language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French", "German"])

# Process uploaded file and save to the vector store
def process_uploaded_file(uploaded_file):
    # Save the uploaded file temporarily to the `TEXTS_DIRECTORY`
    if not os.path.exists(TEXTS_DIRECTORY):
        os.makedirs(TEXTS_DIRECTORY)
    
    # Save the uploaded file to the directory
    file_path = os.path.join(TEXTS_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and chunk the newly uploaded document into the vector store
    load_and_chunk_documents_to_vector_store(file_path)

# Document upload option
uploaded_file = st.sidebar.file_uploader("Upload a text document", type="txt")
if uploaded_file:
    st.sidebar.write("Document uploaded successfully!")
    process_uploaded_file(uploaded_file)

# Feedback/Rating System
st.sidebar.markdown("### Rate the Answer")
rating = st.sidebar.radio("Rate the response quality:", options=["👍 Excellent", "👌 Good", "👎 Poor"])

# Initialize flags in session_state if not already set
if "model_info_displayed" not in st.session_state:
    st.session_state.model_info_displayed = False
if "chat_info_reset" not in st.session_state:
    st.session_state.chat_info_reset = False
if "chat_history_downloaded" not in st.session_state:
    st.session_state.chat_history_downloaded = False

# Show Model Info button
if st.sidebar.button("Show Model Info"):
    st.session_state.chat_info_reset = True
    st.sidebar.markdown(f"**Model Name**: {selected_model_name}")
    st.sidebar.markdown(f"**Max Tokens**: {max_tokens}")
    st.sidebar.markdown(f"**Temperature**: {temperature}")
    st.sidebar.markdown(f"**Top-P**: {top_p}")

# Allow User to Reset Chat History
if st.sidebar.button("Reset Chat History"):
    st.session_state.chat_info_reset = False
    st.session_state.chat_history.clear()
    st.session_state.all_topics.clear()
    st.write("Chat history has been reset.")

if st.sidebar.button("Download Chat History"):
    st.session_state.chat_history_downloaded = True
    chat_history_str = "\n".join([f"You: {entry['user']}\nAI: {entry['bot']}" for entry in st.session_state.chat_history])
    st.download_button("Download as .txt", chat_history_str, file_name="chat_history.txt")

# Initialize LLM pipelines for different models
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Load the selected model
tokenizer, model = load_model(model_dict[selected_model_name])

# Initialize the query pipeline with the user-defined settings
query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=float16,
    max_new_tokens=max_tokens,  # Pass max_tokens here
    temperature=temperature,    # Pass temperature here
    top_p=top_p,                # Pass top_p here
    device="cpu",               # Explicitly use CPU
    eos_token_id=tokenizer.eos_token_id,
)

qa_model = HuggingFacePipeline(pipeline=query_pipeline)

# RetrievalQA chain setup
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=qa_model,
    retriever=vector_store.as_retriever()
)

# Streamlit UI
st.title("Teach Me Anything!")
st.markdown("Streamlit RAG with LangChain and ChromaDB: A Retrieval-Augmented Generation (RAG) app with enhanced prompt engineering.")

# Ensure session state is initialized for all necessary variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize with an empty list

# Input box
user_input = st.text_input("Ask a question:", "")

# Add condition to check if topic has changed or sidebar is updated, only then process user input
if user_input and not st.session_state.topic_changed and not st.session_state.model_info_displayed and not st.session_state.chat_info_reset and not st.session_state.chat_history_downloaded:
    # Retrieve documents
    retrieved_documents = vector_store.similarity_search(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_documents])

    # Refined prompt
    prompt = PromptTemplate.from_template(
        template="""
        You are an AI assistant specialized in {topic}. Your goal is to provide clear, concise, accurate, and complete answers. 
        Use the context provided to generate a well-informed response.
        
        Context: {context}
        Question: {question}
        
        Follow these principles:
        1. Keep the answer under 50 words.
        2. Use simple, professional language.
        3. Provide factual, unbiased information.
        Answer:"""
    ).format(topic=st.session_state.current_topic, context=context, question=user_input)

    # Generate response
    result = qa_model(prompt=prompt)
    answer = result.split("Answer:")[-1].strip()
    answer = answer.split(".")[0] + "."

    # Display answer
    st.session_state.chat_history.append({"user": user_input, "bot": answer})
    st.write(f"**AI:** {answer}")

elif st.session_state.topic_changed:
    st.write("**Note:** The topic has been changed. Please ask a question to continue.")

# Display chat history
st.write("### Chat History")
for entry in st.session_state.chat_history:
    st.write(f"**You:** {entry['user']}")
    st.write(f"**AI:** {entry['bot']}")