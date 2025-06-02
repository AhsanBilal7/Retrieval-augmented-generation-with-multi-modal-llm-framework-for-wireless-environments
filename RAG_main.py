# %pip install -U --quiet langchain langchain-chroma langchain-community openai langchain-experimental
# %pip install --quiet "unstructured[all-docs]" pypdf pillow pydantic lxml pillow matplotlib chromadb tiktoken
# %pip install sentence-transformers scikit-learn tqdm

import csv
import os
import base64
import io
import re
import uuid
import pickle
import time
from matplotlib import pyplot as plt
from time import sleep
from tqdm import tqdm

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatVertexAI
from langchain_google_vertexai import VertexAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings

# Additional imports
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MultiModalRAGPipeline:
    """Multi-Modal RAG Pipeline for Wireless Environment Perception"""
    
    def __init__(self, openai_api_key=None, google_api_key=None):
        self.openai_api_key = openai_api_key or "YOUR_OPENAI_API_KEY"
        self.google_api_key = google_api_key or "YOUR_GOOGLE_API_KEY"
        self.vectorstore = None
        self.retriever = None
        
    def load_pdf_documents(self, pdf_path):
        """Load PDF documents and extract text"""
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        texts = [d.page_content for d in docs]
        return texts
    
    def make_extracted_information(self, description_data, angle_data, distance_data, front_yolo_data, back_yolo_data):
        """Create formatted information string from sensor data"""
        # You can modify this function to use different combinations:
        # GPS only, CAM only, GPS+CAM, GPS+YOLO, CAM+YOLO, or all combined
        
        extract_information = (
            f'GPS: The unit 2 (Transmitter) is {float(distance_data):.2f} m away '
            f'at a bearing of {float(angle_data):.2f} degrees from the unit 1 receiver\n'
            f'CAM: {description_data}. For Total number of object in the area. '
            f'In the front: {front_yolo_data}\nIn the back: {back_yolo_data}'
        )
        return extract_information

    def load_csv_column(self, file_path):
        """Load specific columns from CSV file"""
        description_data = []
        angle_data = []
        distance_data = []
        front_yolo_data = []
        back_yolo_data = []
        
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                description_data.append(row["Description"])
                angle_data.append(row["Bearing (degrees)"])
                distance_data.append(row["Distance (m)"])
                front_yolo_data.append(row["description_yolo"])
                back_yolo_data.append(row["description_yolo_back"])
        
        return description_data, angle_data, distance_data, front_yolo_data, back_yolo_data

    def prepare_text_data(self, csv_path, limit=30):
        """Prepare text data from CSV for processing"""
        description_data, angle_data, distance_data, front_yolo_data, back_yolo_data = self.load_csv_column(csv_path)
        
        title_text = "Title: Multi-Model LLM framework for wireless environment perception\n"
        
        total_text = []
        for i, desc_data in enumerate(description_data):
            if i > limit:
                continue
            combined_info = self.make_extracted_information(
                description_data[i], angle_data[i], distance_data[i], 
                front_yolo_data[i], back_yolo_data[i]
            )
            total_text.append(title_text + "\n\n" + combined_info)
        
        return total_text

    def generate_text_summaries(self, texts, tables, summarize_texts=False):
        """Generate summaries of text elements for retrieval optimization"""
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
        
        prompt = PromptTemplate.from_template(prompt_text)
        
        llm = ChatOpenAI(
            temperature=0.15,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key
        )

        summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()
        text_summaries = []
        table_summaries = []

        if texts and summarize_texts:
            print("Summarizing texts...")
            for text in tqdm(texts, desc="Text Summaries", ncols=100):
                sleep(0.1)
                summary = summarize_chain.invoke({"element": text})
                text_summaries.append(summary)
        elif texts:
            text_summaries = texts

        return text_summaries, table_summaries

    def load_divided_summaries_from_file(self, file_path):
        """Load text and table summaries from file with start/end markers"""
        with open(file_path, 'r') as file:
            lines = file.readlines()

        text_summaries = []
        table_summaries = []
        current_summary = []
        reading_text_summary = False
        reading_table_summary = False

        for line in lines:
            line = line.strip()

            if line == "Text Summary Start":
                reading_text_summary = True
                continue
            elif line == "Text Summary End":
                text_summaries.append("\n".join(current_summary))
                current_summary = []
                reading_text_summary = False
                continue
            elif line == "Table Summary Start":
                reading_table_summary = True
                continue
            elif line == "Table Summary End":
                table_summaries.append("\n".join(current_summary))
                current_summary = []
                reading_table_summary = False
                continue

            if reading_text_summary or reading_table_summary:
                current_summary.append(line)

        return text_summaries, table_summaries

    def encode_image(self, image_path):
        """Convert image to base64 encoded string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_summarize(self, img_base64, prompt):
        """Generate image summary using Gemini model"""
        model = GoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=self.google_api_key, 
            temperature=0.15
        )

        msg = model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ])
        ])
        return msg.content

    def generate_img_summaries(self, image_path):
        """Generate summaries and base64 encoded strings for images"""
        img_base64_list = []
        image_summaries = []

        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

        for img_file in sorted(os.listdir(image_path)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(image_path, img_file)
                base64_image = self.encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(self.image_summarize(base64_image, prompt))

        return img_base64_list, image_summaries

    def create_multi_vector_retriever(self, vectorstore, text_summaries, texts, 
                                    table_summaries, tables, image_summaries, images):
        """Create retriever that indexes summaries but returns raw content"""
        store = InMemoryStore()
        id_key = "doc_id"

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

        # Add documents to retriever
        if text_summaries:
            add_documents(retriever, text_summaries, texts)
        if table_summaries:
            add_documents(retriever, table_summaries, tables)
        if image_summaries:
            add_documents(retriever, image_summaries, images)

        return retriever

    def setup_vectorstore(self, persist_directory="./chroma_db", collection_name="mm_rag_pipeline"):
        """Setup Chroma vectorstore with HuggingFace embeddings"""
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )
        return self.vectorstore

    def save_retriever(self, retriever, docstore_file="docstore.pkl"):
        """Save the retriever's docstore for later use"""
        with open(docstore_file, 'wb') as f:
            pickle.dump(retriever.docstore, f)
        print(f"InMemoryStore (docstore) saved to {docstore_file}.")

    def looks_like_base64(self, sb):
        """Check if string looks like base64"""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

    def is_image_data(self, b64data):
        """Check if base64 data represents an image"""
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def resize_base64_image(self, base64_string, size=(128, 128)):
        """Resize base64 encoded image"""
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        resized_img = img.resize(size, Image.LANCZOS)
        
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def split_image_text_types(self, docs):
        """Split base64-encoded images and texts"""
        b64_images = []
        texts = []
        
        for doc in docs:
            if isinstance(doc, Document):
                doc = doc.page_content
            if self.looks_like_base64(doc) and self.is_image_data(doc):
                doc = self.resize_base64_image(doc, size=(1300, 600))
                b64_images.append(doc)
            else:
                texts.append(doc)
        
        if len(b64_images) > 0:
            return {"images": b64_images[:1], "texts": []}
        return {"images": b64_images, "texts": texts}

    def img_prompt_func(self, data_dict):
        """Create prompt for multi-modal analysis"""
        formatted_texts = "\n".join(data_dict["context"]["texts"])
        messages = []

        text_message = {
            "type": "text",
            "text": (
                "You are an expert analyst tasked with providing detailed analysis.\n"
                "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
                "Use this information to provide comprehensive analysis related to the user question.\n"
                f"User-provided question: {data_dict['question']}\n\n"
                "Text and / or tables:\n"
                f"{formatted_texts}"
            ),
        }
        messages.append(text_message)

        if data_dict["context"]["images"]:
            for image in data_dict["context"]["images"]:
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                messages.append(image_message)
        
        return [HumanMessage(content=messages)]

    def create_rag_chain(self, retriever):
        """Create multi-modal RAG chain"""
        model = GoogleGenerativeAI(
            model="gemini-1.0-pro", 
            google_api_key=self.google_api_key
        )
        
        chain = (
            {
                "context": retriever | RunnableLambda(self.split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.img_prompt_func)
            | model
            | StrOutputParser()
        )
        
        return chain

    def preprocess_response(self, response):
        """Clean up response formatting"""
        return response.strip()

class RAGEvaluator:
    """Evaluation metrics for RAG system performance"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def calculate_cosine_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        vectorizer = CountVectorizer().fit([text1, text2])
        vec1 = vectorizer.transform([text1]).toarray()
        vec2 = vectorizer.transform([text2]).toarray()
        return cosine_similarity(vec1, vec2)[0][0]
    
    def calculate_f1(self, result, ground_truth):
        """Calculate F1 score based on token overlap"""
        result_tokens = set(result.split())
        ground_truth_tokens = set(ground_truth.split())
        
        if len(result_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 0
        
        common_tokens = result_tokens & ground_truth_tokens
        precision = len(common_tokens) / len(result_tokens)
        recall = len(common_tokens) / len(ground_truth_tokens)
        
        if precision + recall == 0:
            return 0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_correctness(self, result, ground_truth, omega=0.25):
        """Calculate overall correctness score"""
        cos_sim = self.calculate_cosine_similarity(result, ground_truth)
        f1 = self.calculate_f1(result, ground_truth)
        return omega * cos_sim + (1 - omega) * f1
    
    def calculate_faithfulness(self, result, ground_truth):
        """Calculate faithfulness score"""
        result_tokens = set(result.split())
        ground_truth_tokens = set(ground_truth.split())
        
        if len(result_tokens) == 0:
            return 0
        
        common_tokens = result_tokens & ground_truth_tokens
        return len(common_tokens) / len(result_tokens)
    
    def evaluate_with_sentence_transformers(self, text1, text2):
        """Evaluate similarity using sentence transformers"""
        embedding_1 = self.sentence_model.encode(text1, convert_to_tensor=True)
        embedding_2 = self.sentence_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding_1, embedding_2)

# Example usage
def main():
    # Initialize the pipeline
    pipeline = MultiModalRAGPipeline(
        openai_api_key="YOUR_OPENAI_API_KEY",
        google_api_key="YOUR_GOOGLE_API_KEY"
    )
    
    # Setup paths (replace with your actual file paths)
    pdf_path = "./path/to/your/document.pdf"
    csv_path = "./path/to/your/knowledge_base_description.csv"
    summaries_path = "./path/to/your/summaries.txt"
    image_path = "./path/to/your/images/"
    
    # Load and prepare data
    texts = pipeline.prepare_text_data(csv_path, limit=30)
    text_summaries, table_summaries = pipeline.load_divided_summaries_from_file(summaries_path)
    
    # Setup vectorstore and create retriever
    vectorstore = pipeline.setup_vectorstore()
    
    # Process images (if available)
    img_base64_list, image_summaries = [], []
    if os.path.exists(image_path):
        img_base64_list, image_summaries = pipeline.generate_img_summaries(image_path)
    
    # Create retriever
    retriever = pipeline.create_multi_vector_retriever(
        vectorstore, text_summaries, texts, 
        table_summaries, [], image_summaries, img_base64_list
    )
    
    # Save retriever
    pipeline.save_retriever(retriever)
    
    # Create RAG chain
    rag_chain = pipeline.create_rag_chain(retriever)
    
    # Example query
    query = """Based on the extracted information provide a detailed and accurate description of the physical 
    environment in structured manner. Include estimates of distances, object types, and any potential 
    obstacles between units. Please keep it clear and simple and include all the numerical values."""
    
    # Run inference
    start_time = time.time()
    result = rag_chain.invoke(query)
    end_time = time.time()
    
    result = pipeline.preprocess_response(result)
    
    print(f"Query: {query}")
    print(f"Result: {result}")
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    # Evaluate results (if ground truth is available)
    evaluator = RAGEvaluator()
    
    # Example ground truth - replace with your actual ground truth
    ground_truth = "Your ground truth text here..."
    
    if ground_truth:
        print("\n--- Evaluation Results ---")
        similarity = evaluator.evaluate_with_sentence_transformers(result, ground_truth)
        correctness = evaluator.calculate_correctness(result, ground_truth)
        faithfulness = evaluator.calculate_faithfulness(result, ground_truth)
        
        print(f"Cosine Similarity: {similarity}")
        print(f"Correctness Score: {correctness:.4f}")
        print(f"Faithfulness Score: {faithfulness:.4f}")

if __name__ == "__main__":
    main()