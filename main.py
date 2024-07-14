import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import GooglePalm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama  # Added for ChatOllama model
from langchain_community.retrievers import BM25Retriever  # Added for BM25 retriever
from collections import defaultdict
from typing import Union
from config import GOOGLE_API_KEY

# Set Google API Key
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY  # Ensure GOOGLE_API_KEY is defined in config.py

# Create logger
import logging
logger = logging.getLogger(__name__)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        handle_file_processing_error("PDF", e)
    return text

# Function to handle file processing errors
def handle_file_processing_error(file_type: str, error: Exception):
    st.error(f"Error processing {file_type} file: {error}")
    logger.exception(f"Error processing {file_type} file", exc_info=True)

# Function to validate user input
def validate_user_input(user_input: Union[st.file_uploader, str]):
    if not user_input:
        st.warning("Please provide valid input.")
        return False
    return True

# Function to handle user feedback
def handle_user_feedback(feedback: str):
    st.success("Thank you for your feedback!")

# Function to handle AI model interaction errors
def handle_model_interaction_error(error: Exception):
    st.error(f"Error interacting with AI model: {error}")
    logger.exception("Error interacting with AI model", exc_info=True)

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to classify topics (mock implementation for demonstration)
def classify_topic(chunk):
    # Example: Mock topic classification based on keywords
    if 'introduction' in chunk.lower():
        return 'Introduction'
    elif 'methodology' in chunk.lower():
        return 'Methodology'
    else:
        return 'Other'

# Function to create a hierarchical indexer
class HierarchicalIndexer:
    def __init__(self):
        self.index = defaultdict(list)

    def add_chunk(self, topic, chunk):
        self.index[topic].append(chunk)

    def get_chunks(self, topic):
        return self.index.get(topic, [])

# Function to expand user queries with related terms
def expand_query(query):
    # Mock implementation: Adding 'related' to the query
    return query + " related"

# Function to merge retrieval results
def merge_results(vector_results, keyword_results):
    # Mock implementation: Merge results from vector-based and keyword-based retrievals
    return vector_results + keyword_results

# Function to perform re-ranking based on relevance (mock implementation)
def rerank_results(results):
    # Mock implementation: Sort results based on relevance scores
    return sorted(results, key=lambda x: x['relevance_score'], reverse=True)

# Function to create a vector store using Google Palm embeddings
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding_model=embeddings)
    return vector_store

# Function to create a multi-document retriever with BM25 and vector-based retrieval
class MultiDocumentRetriever:
    def __init__(self, vector_store, indexer):
        self.vector_store = vector_store
        self.indexer = indexer
        self.bm25_retriever = BM25Retriever()

    def retrieve(self, query, topic=None):
        expanded_query = expand_query(query)
        if topic:
            chunks = self.indexer.get_chunks(topic)
            vector_results = self.vector_store.similarity_search(expanded_query, chunks)
            bm25_results = self.bm25_retriever.retrieve(query, chunks)
        else:
            vector_results = self.vector_store.similarity_search(expanded_query)
            bm25_results = self.bm25_retriever.retrieve(query)

        merged_results = merge_results(vector_results, bm25_results)
        return rerank_results(merged_results)

# Function to initialize conversational chain with ChatOllama model in offline mode
def get_conversational_chain_offline(vector_store, indexer):
    sol_model = ChatOllama(model="llama3") # Install ollama and Llama 3 for it to work
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = MultiDocumentRetriever(vector_store, indexer)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=sol_model, retriever=retriever, memory=memory)
    return conversation_chain

# Function to initialize conversational chain with Google Palm model in online mode
def get_conversational_chain_online(vector_store, indexer):
    llm = GooglePalm() #Put your google API key in config.py file 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = MultiDocumentRetriever(vector_store, indexer)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return conversation_chain

# Main function to run SCHOLAR application
def main():
    st.set_page_config(page_title="SCHOLAR 🚀", layout="wide")
    st.header("Semantic Content Hierarchy and Optimized Learning Augmented Retrieval")
    user_question = st.chat_input("Ask Questions about Everything")

    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.files_uploaded = False

    with st.sidebar:
        st.title("SCHOLAR")
        model_mode = st.toggle("Online Mode")

        st.subheader("Upload your Files here")
        files = st.file_uploader("Upload your Files and Click on the NEXT Button", accept_multiple_files=True, type=["pdf"])

        if st.button("NEXT"):
            if validate_user_input(files):
                with st.spinner("Processing your Files..."):
                    raw_text = ""
                    for file in files:
                        try:
                            raw_text += extract_text_from_pdf(file)
                        except Exception as e:
                            handle_file_processing_error(file.name.split(".")[-1].lower(), e)

                    text_chunks = get_text_chunks(raw_text)
                    indexer = HierarchicalIndexer()
                    for chunk in text_chunks:
                        topic = classify_topic(chunk)
                        indexer.add_chunk(topic, chunk)
                    vector_store = get_vector_store(text_chunks)

                    if model_mode:
                        st.session_state.conversation = get_conversational_chain_online(vector_store, indexer)
                    else:
                        st.session_state.conversation = get_conversational_chain_offline(vector_store, indexer)

                    st.session_state.files_uploaded = True
                    st.success("Processing Done!")
            else:
                st.warning("Please upload at least one file.")

    if user_question:
        user_input(user_question)

    if not st.session_state.files_uploaded:
        st.warning("Start the chat by uploading your files.")
    elif st.session_state.files_uploaded and not files:
        st.session_state.files_uploaded = False

# Function to handle user input during conversation
def user_input(user_question):
    if st.session_state.conversation:
        try:
            is_feedback = "@scholar" in user_question.lower()
            if is_feedback:
                handle_user_feedback(user_question)
            else:
                response = st.session_state.conversation({'question': user_question})
                if 'chat_history' in response:
                    st.session_state.chat_history = response['chat_history']
                if st.session_state.chat_history:
                    for message in st.session_state.chat_history:
                        if isinstance(message, HumanMessage):
                            with st.beta_expander("User"):
                                st.write(message.content)
                        elif isinstance(message, AIMessage):
                            with st.beta_expander("AI"):
                                st.write(message.content)
                    
                    # Auto-scroll to the end of the chat
                    st.markdown(
                        """
                        <script>
                        var element = document.getElementById("end-of-chat");
                        element.scrollIntoView({behavior: "smooth"});
                        </script>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            handle_model_interaction_error(e)
            st.error("An error occurred during conversation. Please try again.")
    else:
        st.warning("Please upload files and click 'NEXT' to start the conversation.")

# Run the main function
if __name__ == "__main__":
    main()
