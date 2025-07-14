# 💬 MediChat ViVi – Health Care Chatbot with RAG & Fine-tune
- MediChat: Enhancing Medical Chatbot Accuracy with Fine-tuning and Retrieval-Augmented Generation

MediChat ViVi is a healthcare-support chatbot system built with advanced techniques such as Retrieval-Augmented Generation (RAG), large language model fine-tuning, and a smart combination of both. The goal is to create a virtual assistant that provides accurate, contextual, and trustworthy answers to users.

---

## 📁 Project Structure

```text
📦 MediChat-ViVi  
 ┣ 📂 **P1_RAG_Method**: Implements a QA system based on *Retrieval-Augmented Generation* (RAG).  
 ┣ 📂 **P2_Finetune_Method**: Fine-tunes a large language model (LLM) on health domain-specific data.  
 ┣ 📂 **P3_RAG&Finetune_Method**: Combines both RAG and fine-tuned models for improved accuracy and flexibility.  
 ┗ 📜 README.md
```

---

## 📂 P1_RAG_Method: Basic RAG Implementation

Utilizes Retrieval-Augmented Generation (RAG) to extract information from textual data and provide intelligent responses.

### 🧹 1. Data Processing:

#### ✅ Clean textual data:
- Extracts text from PDFs using `pdfplumber`, processes tables, and removes page numbers.
- Cleans and merges broken lines, standardizes content.

#### ✅ OCR for scanned or unextractable documents:
Uses `PyMuPDF`, `OpenCV`, and `Aspose.OCR` to convert each PDF page into images, extract text via OCR, and save as `.txt` files.

### 📄 2. Chunking
- Uses `RecursiveCharacterTextSplitter` to split documents into meaning-preserving chunks.

### 🧠 3. Vectorization
- Embeds text using `all-MiniLM-L6-v2` to create semantic vectors.

### 🔎 4. Build Pinecone VectorDB
- Indexes and queries data in Pinecone to retrieve relevant chunks for user questions.

### 🤖 5. Integrate with LLM
- Integrates with LLaMA-2 via `CTransformers`.
- Uses `RetrievalQA` to combine query and retrieved documents for final answers.

### 🌐 6. Web Interface
- Built with Flask + HTML.
- Users can ask questions and receive real-time answers through the RAG pipeline.

---

## 📂 P2_Finetune_Method: Fine-tuning LLM with Custom Data

Fine-tunes the language model on medical data to enhance contextual accuracy.

### 🧹 1. Create conversational dataset
Processes data from JSON (disease list, descriptions, symptoms, treatments), and converts it into a dialogue format simulating interactions between user and assistant.

#### 🧩 Conversation structure:
- **System**: Sets AI assistant behavior (helpful, reliable).
- **User**: User’s question.
- **Assistant**: Answer based on medical knowledge.

```json
[
  {"from": "system", "value": "Bạn là một trợ lí Tiếng Việt..."},
  {"from": "human", "value": "Triệu chứng về bệnh Cúm?"},
  {"from": "gpt", "value": "Triệu chứng: ho, sốt, đau đầu..."}
]
```

### 🔧 2. Format using LLaMA 3 standard
- Converts conversations into format compatible with LLaMA 3 tokenizer.

### 🔄 3. Fine-tuning with Unsloth
- Utilized the `LLaMA-3.2-3B-Instruct-Frog` model quantized to 4-bit.
- Configured LoRA with r=8, lora_alpha=16, dropout=0.1, and checkpointing.
- Applied the chat_template from LLaMA-3.1 to standardize the dialogue input format.
- Fine-tuned using LoRA to reduce memory usage and accelerate training speed.

### 📦 4. Training and Saving the Model
- Configured `SFTTrainer` with the AdamW 8-bit optimizer. Integrated gradient accumulation and bf16 for efficient model training.
- Saved the trained model into the `Models/HCM_4BIT00006` directory.

### 🔗 5. Merging the Model
- Merged the fine-tuned model with the base model and saved the tokenizer.

---

## 📂 P3_RAG&Finetune_Method: Combining RAG + Fine-tuning

Leveraged the strengths of both methods to build a more powerful and context-aware chatbot.

### 🧹 1. Data Processing from PDFs
- Extracted, cleaned, and categorized content into: description, causes, mechanism, and meaning.
- Saved the structured data in JSON format.

### 📄 2. Convert to LangChain Document Format
- Transformed the JSON data into `Document` objects compatible with LangChain for use in RAG pipelines.

### 🧠 3. Semantic Chunking
- Applied **SemanticChunker** (LangChain Experimental) to segment the text semantically instead of using raw character-based splits.
- Used the embedding model `BAAI/bge-base-en-v1.5`.

### 🔍 4. BM25  
- Integrated **BM25** as an additional keyword-based retrieval strategy to enhance information recall.

### 🏷️ 5. Reranker  
- Used a **Reranker** to score the relevance between user queries and retrieved chunks, improving result ranking and filtering.

### 🧠 6. Smart Retriever
- Implemented a smart retrieval strategy combining semantic search, BM25, and reranker to return the most relevant context.

### 🤖 7. Using the Fine-tuned Model
- Loaded the fine-tuned model (`4BIT00006`) with 4-bit QLoRA configuration.
- Configured the tokenizer correctly (e.g., `pad_token`, `eos_token`).

### 💬 8. Prompt & Answer Generation
- Constructed prompts using a role-based format (system/user).
- Generated answers from retrieved context using the fine-tuned model with appropriately set sampling parameters.

---

## 📌 Project Goals

- ✅ Provide accurate and context-aware answers  
- ✅ Minimize hallucination from the model  
- ✅ Customize the system specifically for the healthcare domain

---

## 🛠️ Technologies Used

- Python, LangChain, Flask, Pinecone  
- HuggingFace Transformers, Unsloth  
- LLaMA 2 & 3 (4-bit)  
- SemanticChunker, BM25, Reranker  
- pdfplumber, PyMuPDF, OCR tools

---

## 🚀 Future Development

- Integrate with real-time databases (completed in another repository)  
- Apply agentic AI design for specialized task handling using multiple models  
- Deploy on cloud or GPU-based servers  
- Enhance data quality and integrate medical validation

---

## 👨‍💻 Authors

**Đặng Kim Thành** – AI student, HCMC University of Technology and Education  
**Bùi Quốc Khang** – AI student, HCMC University of Technology and Education  
Contact: dangkimthanh281003@gmail.com
