from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
openapi_key= st.secrets["API_KEY"]

# streamlit page configuration
st.set_page_config(page_title="Kore wa GPT")
st.title("AI sensei")

# initialize session state variables
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt']=""
    
if 'generated' not in st.session_state:
    st.session_state['generated']=[]
    
if 'past' not in st.session_state:
    st.session_state['past']=[]   
    
def submit():
    st.session_state.entered_prompt= st.session_state.prompt_input
    st.session_state.prompt_input= ""
    
chat =ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    openai_api_key=openapi_key,
    max_tokens=100
)              

def build_message_list():
    """
    Build a list of messages including system ,human and AI messages
    """
    zipped_messages = [SystemMessage(
        content = """your name is AI Mentor. You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone.

                1. Greet the user politely ask user name and ask how you can assist them with AI-related queries.
                2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and related topics.
                3. you must Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
                4. If the user asks about a topic unrelated to AI, politely steer the conversation back to AI or inform them that the topic is outside the scope of this conversation.
                5. Be patient and considerate when responding to user queries, and provide clear explanations.
                6. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
                7. Do Not generate the long paragarphs in response. Maximum Words should be 100.

                Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritize their learning experience and well-being."""
    )]
    for human_msg, ai_msg in zip_longest(st.session_state['past'],st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg
            ))
        if ai_msg is not None:
            zipped_messages.append(AIMessage(
                content=ai_msg
            ))   
    return zipped_messages 

def generate_response():
    """
    Generate AI response using the chatgpt openai model.
    """
    zipped_messages= build_message_list()
    ai_response= chat(zipped_messages)
    response=ai_response.content
    return response

st.text_input("kimi:",key="prompt_input",on_change=submit)

if st.session_state.entered_prompt !="":
    user_query= st.session_state.entered_prompt
    st.session_state.past.append(user_query)
    output= generate_response()
    st.session_state.generated.append(output)
    
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, 1,-1):
        message(st.session_state["generated"][i],key=str(i))
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')
            