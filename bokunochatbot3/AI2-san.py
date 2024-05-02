import streamlit as st 
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import VectorStoreIndex,ServiceContext,Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

openai.openai_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="LLAMA no Chikara",layout="centered",initial_sidebar_state="auto",menu_items=None)
st.title("Hanashi Mashou...")

if "messages" not in st.session_state.keys():
    st.session_state.messages= [{"role":"assistant","content":"Ask me about Pakistan Studies!"}]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and Indexing the Documnent..."):
        reader= SimpleDirectoryReader(input_dir="./data",recursive=True)
        docs=reader.load_data()
        emb_model= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm=OpenAI(model="gpt-3.5-turbo",temperature="0.5", systemprompt="You are expert on the Pakistan studies. Your job is to provide the valid and relevant answers.Assuming all the queries related to Pakistan Studies. Keep your answers based on facts .Do not hallucinate")
        service_content= ServiceContext.from_defaults(llm=llm,embed_model=emb_model)
        index= VectorStoreIndex.from_documents(docs,service_context=service_content)
        return index

index= load_data()
chat_engine= index.as_chat_engine(chat_mode="condense_question",verbose=True)

if prompt :=st.chat_input("Your Question"):
    st.session_state.messages.append({"role":"user","content":prompt})
        
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])    
        
if st.session_state.messages[-1]["role"] !="assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response= chat_engine.chat(prompt)
            message={"role":"assistant","content":response.response}
            st.session_state.messages.append(message)    
            
               