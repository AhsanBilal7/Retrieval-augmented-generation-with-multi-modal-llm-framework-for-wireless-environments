# Retrieval augmented generation with multi-modal llm framework for wireless environments

A comprehensive Retrieval-Augmented Generation (RAG) system that combines text, images, and sensor data to provide intelligent analysis of wireless communication environments. This pipeline integrates GPS data, camera descriptions, YOLO object detection results, and PDF documents to generate contextual responses about physical environments.

## 🎉✅ **Accepted at ICC 2025 Workshop!**

We're thrilled to announce that our paper has been **accepted** at the **ICC 2025 Workshop**! 🥳📚

🔗 [**Read the paper on arXiv**](https://arxiv.org/pdf/2503.07670)

## 🌟 Features

- **Multi-Modal Data Processing**: Handles text, images, tables, and sensor data
- **Advanced Retrieval**: Uses ChromaDB with HuggingFace embeddings for efficient similarity search
- **LLM Integration**: Supports OpenAI GPT and Google Gemini models
- **Comprehensive Evaluation**: Built-in metrics for correctness, faithfulness, and similarity scoring
- **Flexible Architecture**: Modular design for easy customization and extension
- **Sensor Data Integration**: Processes GPS coordinates, bearing angles, distances, and object detection data

## 🛠️ Installation

### Prerequisites

```bash
pip install -U --quiet langchain langchain-chroma langchain-community openai langchain-experimental
pip install --quiet "unstructured[all-docs]" pypdf pillow pydantic lxml matplotlib chromadb tiktoken
pip install sentence-transformers scikit-learn tqdm
```

### Required Dependencies

- `langchain` and related packages for LLM orchestration
- `chromadb` for vector storage
- `sentence-transformers` for embedding generation
- `scikit-learn` for evaluation metrics
- `PIL` for image processing
- `pandas` for data manipulation

## 📁 Project Structure

```
project/
├── main.py                 # Main pipeline implementation
├── README.md              # This file
├── data/
│   ├── document.pdf       # Your PDF documents
│   ├── knowledge_base_description.csv  # Sensor data CSV
│   ├── summaries.txt      # Pre-generated summaries
│   └── images/            # Directory for image files
├── chroma_db/             # Vector database storage
└── outputs/
    └── docstore.pkl       # Serialized document store
```

## 🚀 Quick Start

### 1. Setup API Keys

```python
# Set your API keys
OPENAI_API_KEY = "your_openai_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
```

### 2. Prepare Your Data

**CSV Format** (knowledge_base_description.csv):
```csv
Description,Bearing (degrees),Distance (m),description_yolo,description_yolo_back
"Urban street with vehicles",183.68,12.03,"24 cars detected","17 cars detected"
```

**Summaries File** (summaries.txt):
```
Text Summary Start
Summary of your text content here...
Text Summary End

Table Summary Start
Summary of your table content here...
Table Summary End
```

### 3. Basic Usage

```python
from main import MultiModalRAGPipeline, RAGEvaluator

# Initialize pipeline
pipeline = MultiModalRAGPipeline(
    openai_api_key="YOUR_OPENAI_API_KEY",
    google_api_key="YOUR_GOOGLE_API_KEY"
)

# Setup data paths
pdf_path = "./data/document.pdf"
csv_path = "./data/knowledge_base_description.csv"
summaries_path = "./data/summaries.txt"
image_path = "./data/images/"

# Load and process data
texts = pipeline.prepare_text_data(csv_path, limit=30)
text_summaries, table_summaries = pipeline.load_divided_summaries_from_file(summaries_path)

# Setup vector store
vectorstore = pipeline.setup_vectorstore()

# Create retriever and RAG chain
retriever = pipeline.create_multi_vector_retriever(
    vectorstore, text_summaries, texts, 
    table_summaries, [], [], []
)

rag_chain = pipeline.create_rag_chain(retriever)

# Query the system
query = "Describe the wireless environment and potential obstacles"
result = rag_chain.invoke(query)
print(result)
```

## 📊 Data Format Requirements

### CSV Data Structure

Your CSV file should contain the following columns:

- `Description`: Textual description of the environment
- `Bearing (degrees)`: Directional bearing from receiver to transmitter
- `Distance (m)`: Distance between units in meters
- `description_yolo`: Object detection results for front camera
- `description_yolo_back`: Object detection results for back camera

### Image Requirements

- Format: JPG files
- Naming: Sequential or descriptive filenames
- Location: Place in designated images directory

## 🔧 Configuration Options

### Model Selection

```python
# Use different LLM models
pipeline = MultiModalRAGPipeline(
    openai_api_key="YOUR_KEY",
    google_api_key="YOUR_KEY"
)

# Modify the model in create_rag_chain method:
# - "gpt-3.5-turbo" or "gpt-4" for OpenAI
# - "gemini-1.0-pro" or "gemini-1.5-flash" for Google
```

### Embedding Options

```python
# Change embedding model in setup_vectorstore:
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Or use OpenAI embeddings:
embedding_function = OpenAIEmbeddings(openai_api_key="YOUR_KEY")
```

### Processing Limits

```python
# Adjust data processing limits
texts = pipeline.prepare_text_data(csv_path, limit=50)  # Process more entries
```

## 📈 Evaluation Metrics

The system includes comprehensive evaluation capabilities:

```python
evaluator = RAGEvaluator()

# Calculate various metrics
similarity = evaluator.evaluate_with_sentence_transformers(result, ground_truth)
correctness = evaluator.calculate_correctness(result, ground_truth)
faithfulness = evaluator.calculate_faithfulness(result, ground_truth)

print(f"Similarity: {similarity}")
print(f"Correctness: {correctness:.4f}")
print(f"Faithfulness: {faithfulness:.4f}")
```

### Available Metrics

- **Cosine Similarity**: Semantic similarity between generated and ground truth text
- **Correctness Score**: Combination of cosine similarity and F1 score
- **Faithfulness Score**: How well the response stays true to the source material
- **F1 Score**: Precision and recall based on token overlap

## 🎯 Use Cases

- **Wireless Communication Analysis**: Analyze RF propagation environments
- **Smart City Planning**: Understand urban infrastructure impact on connectivity
- **Autonomous Vehicle Navigation**: Process sensor data for path planning
- **Environmental Monitoring**: Combine multiple data sources for comprehensive analysis
- **Research Applications**: Multi-modal data analysis for academic research

## 🔍 Example Queries

```python
# Environment analysis
query = "Describe the physical environment and potential signal obstacles"

# Distance and bearing analysis  
query = "What is the distance and bearing between the transmitter and receiver?"

# Object detection summary
query = "Summarize the objects detected in front and back cameras"

# Comprehensive analysis
query = "Provide a detailed analysis combining GPS, camera, and sensor data"
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are valid and have sufficient quota
2. **File Path Issues**: Verify all file paths exist and are accessible
3. **Memory Issues**: Reduce the data processing limit if running out of memory
4. **ChromaDB Errors**: Delete the `chroma_db` directory to reset the vector store

### Performance Optimization

- Use smaller embedding models for faster processing
- Limit the number of documents processed simultaneously
- Implement batch processing for large datasets
- Consider using GPU acceleration for embedding generation

## 📚 Documentation

### Class Overview

- `MultiModalRAGPipeline`: Main orchestration class
  - Data loading and preprocessing
  - Vector store management
  - RAG chain creation
  - Multi-modal processing

- `RAGEvaluator`: Evaluation and metrics class
  - Similarity calculations
  - Performance scoring
  - Comparative analysis

### Key Methods

- `prepare_text_data()`: Process CSV sensor data
- `generate_text_summaries()`: Create optimized summaries
- `create_multi_vector_retriever()`: Setup retrieval system
- `create_rag_chain()`: Build the complete RAG pipeline

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### 📖 Citation (BibTeX)

```bibtex
@article{mohsin2025retrieval,
  title={Retrieval augmented generation with multi-modal llm framework for wireless environments},
  author={Mohsin, Muhammad Ahmed and Bilal, Ahsan and Bhattacharya, Sagnik and Cioffi, John M},
  journal={arXiv preprint arXiv:2503.07670},
  year={2025}
}
```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

## 🚧 Work in Progress

**Note**: This repository is currently under active development and will be cleaned up further. The codebase is a work in progress, and we are continuously improving the structure, documentation, and functionality. Expect regular updates and refinements as we enhance the system's capabilities and user experience.

---

*This project demonstrates the integration of multiple AI technologies for comprehensive environmental analysis and is suitable for research, development, and production applications in wireless communication and smart environment domains.*