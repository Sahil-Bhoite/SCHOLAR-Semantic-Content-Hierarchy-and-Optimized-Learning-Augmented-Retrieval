# SCHOLAR: Semantic Content Hierarchy and Optimized Learning Augmented Retrieval

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Implementation Details](#implementation-details)
7. [Configuration](#configuration)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

## Introduction

SCHOLAR (Semantic Content Hierarchy and Optimized Learning Augmented Retrieval) is an advanced question-answering system designed to process and analyze textbook content. It creates a hierarchical knowledge base for efficient information retrieval and learning augmentation, leveraging state-of-the-art natural language processing techniques to provide accurate and context-aware responses to user queries.

The system is built to handle multiple PDF textbooks, create a semantic understanding of their content, and provide an interactive interface for users to ask questions and receive informed answers based on the processed textbooks.

## Features

1. **PDF Text Extraction**: 
   - Utilizes PyPDF2 to extract text content from PDF files.
   - Handles multiple PDF uploads simultaneously.

2. **Hierarchical Indexing**: 
   - Implements a custom `HierarchicalIndexer` class to organize extracted content into a topic-based structure.
   - Enables efficient retrieval of relevant content based on topics.

3. **Text Chunking**: 
   - Employs `RecursiveCharacterTextSplitter` from langchain to split large texts into manageable chunks.
   - Ensures optimal processing and retrieval of information.

4. **Multi-Document Retrieval**: 
   - Implements a hybrid retrieval system combining vector-based and keyword-based search methods.
   - Enhances the accuracy and relevance of retrieved information.

5. **Query Expansion**: 
   - Includes a query expansion function to enhance user queries (currently a mock implementation).
   - Improves retrieval accuracy by considering related terms and concepts.

6. **BM25 Retrieval**: 
   - Utilizes the BM25 algorithm for keyword-based document retrieval using `BM25Retriever` from langchain.
   - Provides efficient and accurate keyword-based search capabilities.

7. **Vector-based Similarity Search**: 
   - Employs FAISS for efficient similarity search using dense vector representations.
   - Enables semantic search capabilities for more nuanced query understanding.

8. **Result Re-ranking**: 
   - Implements a basic re-ranking system to prioritize the most relevant results.
   - Enhances the quality of responses by presenting the most pertinent information first.

9. **Conversational AI**: 
   - Online mode: Uses Google Palm for generating responses.
   - Offline mode: Uses ChatOllama (Sol model) for generating responses.
   - Provides flexibility in model choice based on connectivity and performance requirements.

10. **User-friendly Interface**: 
    - Built with Streamlit for an intuitive and interactive user experience.
    - Offers easy file upload, mode selection, and question input functionalities.

11. **Error Handling and Logging**: 
    - Implements robust error handling and logging system for improved reliability and debugging.
    - Enhances system stability and facilitates troubleshooting.

## System Architecture

SCHOLAR follows a modular architecture:

1. **Input Processing Module**: Handles PDF upload and text extraction.
2. **Indexing Module**: Creates and manages the hierarchical index of extracted content.
3. **Retrieval Module**: Implements hybrid retrieval combining BM25 and vector-based search.
4. **Query Processing Module**: Handles query expansion and processing.
5. **AI Interaction Module**: Manages interactions with AI models (Google Palm or ChatOllama).
6. **User Interface Module**: Streamlit-based interface for user interactions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SCHOLAR.git
   cd SCHOLAR
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up Google API key:
   - Create `config.py` in the project root.
   - Add your Google API key:
     ```python
     GOOGLE_API_KEY = "your_google_api_key_here"
     ```

5. Run the application:
   ```
   streamlit run main.py
   ```

## Usage

1. Launch the Streamlit app using the command above.
2. In the sidebar:
   - Toggle between online (Google Palm) and offline (ChatOllama) modes.
   - Upload PDF files (textbooks) for processing.
   - Click "NEXT" to process uploaded files.
3. Once processing is complete, use the main chat interface to ask questions.
4. The system will retrieve relevant information and generate responses based on the processed textbooks.

## Implementation Details

### PDF Processing
- `extract_text_from_pdf`: Extracts text from PDF files using PyPDF2.
- `get_text_chunks`: Splits extracted text into manageable chunks using RecursiveCharacterTextSplitter.

### Indexing
- `HierarchicalIndexer`: Custom class for organizing content into a topic-based structure.
- `classify_topic`: (Mock) function to classify text chunks into topics.

### Retrieval
- `MultiDocumentRetriever`: Combines vector-based (FAISS) and keyword-based (BM25) retrieval methods.
- `get_vector_store`: Creates a FAISS vector store using Google Palm embeddings.
- `expand_query`: (Mock) function for query expansion.
- `merge_results`: Combines results from different retrieval methods.
- `rerank_results`: Re-ranks retrieved results based on relevance.

### AI Models
- `get_conversational_chain_online`: Sets up the conversational chain using Google Palm.
- `get_conversational_chain_offline`: Sets up the conversational chain using ChatOllama.

### User Interface
- Streamlit-based interface with sidebar for file upload and mode selection.
- Main area for displaying conversation history and input for user questions.

## Configuration

- `config.py`: Stores the Google API key. Ensure this file is kept secure and not committed to version control.
- Adjust `chunk_size` and `chunk_overlap` in `get_text_chunks` function to optimize text splitting.
- Modify `classify_topic` function to implement more sophisticated topic classification.

## Performance Considerations

- Large PDF files may require significant processing time. Consider implementing a progress bar for file processing.
- Vector stores can consume substantial memory. For large datasets, consider using disk-based storage options.
- Implement caching mechanisms to store processed data and reduce repeated computations.

## Troubleshooting

- Ensure all dependencies are correctly installed and up to date.
- Check `config.py` for correct API key configuration.
- For PDF processing issues, ensure PyPDF2 is compatible with your PDF files.
- If encountering memory issues, try processing smaller chunks of text or use a machine with more RAM.

## Contributing

Contributions to SCHOLAR are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please ensure your code adheres to the project's coding standards and include appropriate tests for new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Langchain community for providing essential NLP tools and frameworks.
- FAISS team for their efficient similarity search implementation.
- Streamlit team for their user-friendly web application framework.
- Google Palm and ChatOllama teams for their powerful language models.
- PyPDF2 developers for the PDF processing capabilities.
- All open-source contributors whose libraries are used in this project.

