from src.helper import *
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_txt("data/")
chunks = text_split(extracted_data)
embedding_model = download_hugging_face_embeddings()


#Initializing the Pinecone
# pinecone.init(api_key=PINECONE_API_KEY,
#               environment=PINECONE_API_ENV)


# index_name="medchat"

# #Creating Embeddings for Each of The Text Chunks & storing
# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
# Khởi tạo đối tượng Pinecone
index_name = 'medchat'
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

db = PineconeVectorStore.from_documents(documents=chunks, embedding=embedding_model, index_name="medchat",
                                            namespace="medchat-vectors")