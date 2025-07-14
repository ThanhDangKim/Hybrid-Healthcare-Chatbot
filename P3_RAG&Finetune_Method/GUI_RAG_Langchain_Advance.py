import streamlit as st
from threading import Thread
from transformers import TextIteratorStreamer
import torch
import time
import os 
from transformers import AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from FlagEmbedding import FlagReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from langchain.schema import Document
import json


# Đọc tệp JSON
def load_json_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Chuyển dữ liệu từ JSON thành định dạng Document của LangChain
def convert_json_to_documents(json_data):
    documents = []

    # Duyệt qua các disease trong dữ liệu JSON
    for disease in json_data.get("diseases", []):
        title = disease.get("title", "")
        description = disease.get("description", "")
        causes = disease.get("causes", "")
        mechanism = disease.get("mechanism", "")
        meaning = disease.get("meaning", "")

        # Tạo content và metadata cho mỗi Document
        content = description + "\n" + causes + "\n" + mechanism + "\n" + meaning
        metadata = {
            "title": title,
            "table": disease.get("tables", [])
        }

        # Tạo Document
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    return documents

# Hàm để load tất cả các tệp JSON trong thư mục
def load_all_json_files_in_directory(directory_path):
    documents = []
    # Duyệt qua tất cả các tệp trong thư mục
    for filename in os.listdir(directory_path):
        # Kiểm tra nếu tệp có phần mở rộng là .json
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory_path, filename)
            # Đọc và chuyển tệp JSON thành documents
            json_data = load_json_data(json_file_path)
            documents.extend(convert_json_to_documents(json_data))
            print(f"Loaded: {filename}")
    return documents

def load_chunk(documents_path):
    # Tải lại tất cả các Document từ tệp pickle
    with open(documents_path, "rb") as f:
        list_of_chunks = pickle.load(f)
    
    return list_of_chunks

def load_embedding_model():
    model_name = "BAAI/bge-base-en-v1.5"
    # model_kwargs = {'device': 'cpu'}
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False} # Chuẩn hóa vector = True sẽ limit length vector = 1

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    return hf_embeddings

def load_Faiss_index(hf_embeddings, db_path):
    index = faiss.IndexFlatL2(len(hf_embeddings.embed_query("hello world")))
    vector_db_path = db_path

    # Tải lại FAISS index từ file
    vector_store = FAISS.load_local(vector_db_path, hf_embeddings, allow_dangerous_deserialization = True)
    return vector_store

def bm25_retriever(list_of_chunks):
    # Khởi tạo BM25 retriever với tham số tìm kiếm top 10 các kết quả liên quan nhất
    bm25_retriever = BM25Retriever.from_documents(
        list_of_chunks, k = 10
    )
    return bm25_retriever

def reranker():
    # Load the tokenizer for the BAAI/bge-m3 model
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    return reranker

class Retriever:
    def __init__(self, semantic_retriever, bm25_retriever, reranker, documents):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.documents = documents

    def __call__(self,query):
        semantic_results = self.semantic_retriever.similarity_search(
            query,
            k=10,
        )
        bm25_results = self.bm25_retriever.invoke(query)

        content = set()
        retrieval_docs = []

        for result in semantic_results:
            if result.page_content not in content:
                content.add(result.page_content)
                retrieval_docs.append(result)

        for result in bm25_results:
            if result.page_content not in content:
                content.add(result.page_content)
                retrieval_docs.append(result)

        pairs = [[query,doc.page_content] for doc in retrieval_docs]

        scores = self.reranker.compute_score(pairs,normalize = True)

        # Lấy tài liệu nguồn từ phần tử con dựa trên điểm số ngưỡng
        context_1 = []
        context_2 = []
        context = []
        parent_ids = set()
        for i in range(len(retrieval_docs)):
            # Điểm liên quan >= 0.6 sẽ được sử dụng làm kiểu ngữ cảnh 1 (chỉ ra sự liên quan cao hơn đối với truy vấn).
            if scores[i] >= 0.6:
                parent_idx = retrieval_docs[i].metadata['parent']
                if parent_idx not in parent_ids:
                    parent_ids.add(parent_idx)
                    context_1.append(self.documents[parent_idx])

        # Điểm liên quan >= 0.1 sẽ được sử dụng làm kiểu ngữ cảnh 2 (chỉ ra sự liên quan trung bình đến thấp đối với truy vấn).
            elif scores[i] >= 0.1:
                parent_idx = retrieval_docs[i].metadata['parent']
                if parent_idx not in parent_ids:
                    parent_ids.add(parent_idx)
                    context_2.append(self.documents[parent_idx])
        
        if len(context_1) > 0:
            print('Context 1')
            context=context_1
        elif len(context_2) > 0:
            print('Context 2')
            context=context_2
        else:
            # Nếu điểm liên quan < 0.1, điều này chỉ ra rằng không có tài liệu liên quan.
            print('No relevant context')
        return context

def load_model():
    # Đường dẫn tới mô hình và tokenizer
    model_path = "../mlperf_env/mlperf_env/model/4BIT00006"

    # Tham số lượng tử hóa mô hình
    use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant = True, "float16", "nf4", False 
    device_map = {"": 0}
    # Tải tokenizer và mô hình với tham số QLoRA 
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Thiết lập pad token thành eos_token (có 2 trường hợp)
        ## Lựa chọn 1: Thiết lập eos_token thành pad_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token 

        # Hoặc, lựa chọn 2: Thêm một padding token mới
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )

    # model.config.use_cache = False
    # model.config.pretraining_tp = 1
    return model, tokenizer

def generate_response_streaming_from_prompt_format(model, tokenizer, user_input, context):
    system_prompt = f"""
    Bạn là một trợ lí ảo MediChat ViVi nhiệt tình và trung thực. 
    Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn. 
    Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, 
    hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác, vui lòng không chia sẻ thông tin sai lệch."""

    formatted_prompt = f"""
        Câu hỏi của người dùng: {user_input}
        Trả lời câu hỏi dựa vào các thông tin sau: {context}"""
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    attention_mask = (inputs != tokenizer.pad_token_id).long()
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = {
        "inputs": inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "temperature": 1.5,
        "min_length": 30,  
        "use_cache": True,
        "top_p": 0.95,
        "min_p": 0.1,
        "attention_mask" : attention_mask,
    }
    
    # Start the generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the generated text
    for _, new_text in enumerate(streamer):
        if "<|eot_id|>" in new_text:
            new_text = new_text.replace("<|eot_id|>", "")
        print(new_text)
        yield new_text
        time.sleep(0.02)

def main():
    # st.image("./resources/af3e7d8c-283e-4c3a-815f-dd1fa4a2af23.webp")
    st.title("Chat with MediChat ViVi")
    st.write("Nhập văn bản và mô hình sẽ trả lời.")

    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        with st.spinner("Đang tải mô hình..."):
            st.session_state.model, st.session_state.tokenizer = load_model()
    
    # Nhập từ người dùng
    user_input = st.text_area("Nhập câu hỏi của bạn:")
    list_of_chunks = load_chunk("vectorstore/db_document/documents.pkl")
    vector_db_path = 'vectorstore/db_faiss'
    json_directory = "Data/JSON"
    hf_embedding = load_embedding_model()
    retriever = Retriever(semantic_retriever = load_Faiss_index(hf_embedding, vector_db_path), bm25_retriever = bm25_retriever(list_of_chunks), reranker = reranker(), documents = load_all_json_files_in_directory(json_directory))

    if st.button("Gửi"):
        if user_input:
            with st.spinner("Đang xử lý..."):
                user_input = user_input.replace('\n','')
                context = retriever(user_input)
                st.write_stream(generate_response_streaming_from_prompt_format(st.session_state.model, st.session_state.tokenizer, user_input, context))
        else:
            st.error("Vui lòng nhập câu hỏi.")

if __name__ == "__main__":
    main()