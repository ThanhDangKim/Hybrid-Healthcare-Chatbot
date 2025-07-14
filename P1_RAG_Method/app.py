# %%
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# %%
embedding_model = download_hugging_face_embeddings()

class VectorDB:
    def __init__(self):
        self.embedding = embedding_model
        self.db = self.__build_db__()

    def __build_db__(self):
        db = PineconeVectorStore.from_existing_index(
            index_name="medchat",
            namespace="medchat-vectors",
            embedding=self.embedding
        )
        return db
    
    def get_retriever(self,
                      search_type: str = 'similarity', 
                      search_kwargs: dict = {'k': 10}):
        retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        return retriever

#Initializing the Pinecone
# pinecone.init(api_key=PINECONE_API_KEY,
#               environment=PINECONE_API_ENV)

# index_name="medchat"

# #Loading the index
# docsearch=Pinecone.from_existing_index(index_name, embeddings)
index_name = 'medchat'
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)


# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
def create_prompt_template(template):
    prompt = PromptTemplate(template = template, input_variables = ['context', 'question'])
    return prompt

# chain_type_kwargs={"prompt": PROMPT}
def create_qa(llm, prompt):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = VectorDB().get_retriever(),
        return_source_documents = False,
        chain_type_kwargs = {'prompt': prompt}
    )
    return llm_chain

# %%
model_name = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
llm=CTransformers(model=model_name,
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

# %%
# qa=RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True, 
#     chain_type_kwargs=chain_type_kwargs)
prompt = create_prompt_template(prompt_template)
qa_chain = create_qa(llm, prompt)

# %%
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa_chain({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug= True)
# %%
