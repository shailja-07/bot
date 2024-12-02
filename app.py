from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
hf_api_token = os.environ.get("API_TOKEN")

llm = HuggingFaceEndpoint(
    model=model_name,
    token=hf_api_token  
)
prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate information in one complete bullet point for the following topic:\n\n{context}")
        ]
    )
    
    
llm_chain = LLMChain(llm=llm, prompt=prompt)

app = FastAPI()

class Data(BaseModel):
    text: str

@app.get("/")
@app.head("/")
def root():
    return {"message": "Welcome!"}

@app.get('/{name}')
def get_name(name: str):
    return {'hello': f'{name}'}

@app.post("/detail")
def predict(data: Data):
    input_data = data.text.strip()
    
    if input_data:
        info_text=llm_chain.run(input_data)
    else:
        print("enter topic")
    return {"information": info_text} 