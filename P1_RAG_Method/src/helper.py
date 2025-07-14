from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import json

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def read_transform_json_data():
    with open('../data/health-data.json', 'r') as json_file:
        data = json.load(json_file)

    # Mở tệp txt để ghi
    with open('../data/health-data.txt', 'w') as txt_file:
        for disease in data['diseases']:
            name = disease['name']
            description = disease['description']
            symptoms = ', '.join(disease['symptoms'])
            treatments = ', '.join(disease['treatments'])

            # Viết các thông tin vào tệp
            txt_file.write(f"[INST] What is {name}? [/INST] {description}\n")
            txt_file.write(f"[INST] What are the symptoms of {name}? [/INST] Symptoms include {symptoms}.\n")
            txt_file.write(f"[INST] How can {name} be treated? [/INST] Treatments include {treatments}.\n\n")

#Extract data from txt file
def load_txt(data):
    class UTF8TextLoader(TextLoader):
        def __init__(self, file_path):
            super().__init__(file_path, encoding='utf-8')

    read_transform_json_data()
    txt_path = data
    loader = DirectoryLoader(path=txt_path, glob='*.txt', loader_cls=UTF8TextLoader)
    documents = loader.load()
    return documents

#Create text chunks
def text_split(extracted_data):
    separators = [
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ]
    chunk_size = 500
    chunk_overlap = 50

    def word_count(text):
        return len(text.split())
    
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    # text_chunks = text_splitter.split_documents(extracted_data)
    # return text_chunks
    document_splitter = RecursiveCharacterTextSplitter(
        separators = separators,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = word_count,
        is_separator_regex = False
    )
    chunks = document_splitter.split_documents(extracted_data)
    return chunks

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings