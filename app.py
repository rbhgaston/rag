import os
import streamlit as st
from langchain_community.llms import Ollama

###
from create_db import create_db, set_DATA_PATH
from main import get_results, create_prompt, response_llm

# this line solves the issue "ValueError: Could not connect to tenant default_tenant. Are you sure it exists?"
import chromadb 
chromadb.api.client.SharedSystemClient.clear_system_cache()

## SESSION STATE
if "CREATED_DB" not in st.session_state:
    st.session_state["CREATED_DB"] = False

# SIDEPANEL
with st.sidebar:
    
    folder_path = st.text_input('Input folder path')
    set_DATA_PATH(folder_path)

    # FOLDER SLECTOR
    file_paths = []
    if os.path.isdir(folder_path):
        for fn in os.listdir(folder_path):
            fp = f'{folder_path}/{fn}'
            if os.path.isfile(fp):
                file_paths.append(fn)

    # # Select file from scanned folder.
    # selected_file = st.selectbox('Select file', options=file_paths)
    

    # CREATE DB
    button = st.button('Create DB')
    if button:
        with st.status('Creating DB...'):
            create_db()
            st.write('DB created')
            st.session_state["CREATED_DB"] = True

    # # add files 
    # st.button('Add files')


# MAIN PANEL
st.title("💬 RAG")
st.caption("🚀 explore your own files ")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input := st.chat_input():
        
    # creating a chat history
    st.session_state.messages.append({"role": "user", "content": input})
    st.chat_message("user").write(input)

    #### RAG

    results = get_results(input)
    prompt = create_prompt(results, input)
    response = response_llm(prompt, results)
    ###

    msg = response
    # update chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").markdown(msg)




    



