from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import time

model_name = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"

def load_model():
    llm = CTransformers(
        model = model_name,
        model_type = 'llama',
        max_new_token = 1024,
        temparature = 0.01
    )
    return llm

def create_prompt_template(template):
    prompt = PromptTemplate(template = template, input_variables = ['question'])
    return prompt

def create_simple_chain(llm, prompt):
    llm_chain = LLMChain(llm = llm, prompt = prompt)
    return llm_chain

template = '''<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant'''

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

llm = load_model()
prompt = create_prompt_template(prompt_template)
llm_chain = create_simple_chain(llm, prompt)

# question = "What factors increase the risk of developing pneumoconiosis?"
question = "What are the symptoms of pneumonia?"
start_time = time.time()
output = llm_chain.invoke({'question': question})
response_time = time.time() - start_time
print(output)
print(f"Thời gian phản hồi: {response_time:.2f} giây")



