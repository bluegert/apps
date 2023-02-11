# from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import os
# from supabase import create_client, Client
# import asyncio
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from operator import itemgetter
import PyPDF2
from streamlit_chat import message
import base64

openai.api_key = st.secrets['api_key']

def generate_response(prompt):
    one_shot_prompt = '''I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
    Q: What is human life expectancy in the United States?
    A: Human life expectancy in the United States is 78 years.
    Q: '''+prompt+'''
    A: '''
    completions = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = one_shot_prompt,
        max_tokens = 1024,
        n = 1,
        stop=["Q:"],
        temperature=0.2,
    )
    message = completions.choices[0].text
    return message

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

def get_similar_terms(text_input, text_vectors, texts):
    search_term_vector = get_embedding(text_input, engine="text-embedding-ada-002")
    similarities = []
    for i in range(len(text_vectors)):
        similarities.append(cosine_similarity(text_vectors[i], search_term_vector))
    sorted_texts = sorted(list(zip(texts, similarities)),key=itemgetter(1), reverse=True)
    return list(zip(*sorted_texts[:3]))

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
uploaded_file = st.file_uploader("Choose a file first", type="txt")
st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file is not None:
    # if uploaded_file.name.endswith(".pdf"):
    #   base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
    #   pdf_display = (
    #     f'<embed src="data:application/pdf;base64,{base64_pdf}" '
    # 'width="800" height="1000" type="application/pdf"></embed>'
    # )
    #   st.markdown(pdf_display, unsafe_allow_html=True)
    # else:
    texts = text_splitter.split_text(uploaded_file.read().decode("utf-8"))
    text_vectors = []
    for i in range(len(texts)):
      text_vectors.append(get_embedding(texts[i], engine="text-embedding-ada-002"))

    text_input = st.text_input(
        "Ask a question 👇", # make this custom to the pdf
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled
        # placeholder=st.session_state.placeholder,
    )

    if text_input:
      if 'generated' not in st.session_state:
            st.session_state['generated'] = []

      if 'past' not in st.session_state:
          st.session_state['past'] = []
      similar_terms = get_similar_terms(text_input, text_vectors, texts)
      # user_input_embedding_prompt = 'Using this context: "'+similar_terms+'", answer the following question changing as little wording as possible of the context. \n'+ text_input
      # response = generate_response(user_input_embedding_prompt)
      # # if already text generated, build on that
      # if st.session_state['generated']:
      #   st.write(st.session_state['generated'])
      # st.session_state.past.append(text_input)
      # st.session_state.generated.append(response)
      # if st.session_state['generated']:
      #   for i in range(len(st.session_state['generated'])-1):
      #       message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
      #       message(st.session_state["generated"][i], key=str(i))

# pdf support
# https://discuss.streamlit.io/t/how-to-display-pdf-files-in-streamlit/1806/2
# prompts building on each other
# summarizer
# better prompts (see youtube gpt)
# multiple files upload
# making it look nicer
# data privacy
# some metric of performance. Can I give the model feedback? Should I?