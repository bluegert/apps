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

openai.api_key = st.secrets['api_key']

def craft_response(query, msg):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"A journalist just asked you this question: {query}. In the past you have said the following around this topic: {msg}. Craft an insightful response to the journalist based on the facts presented in your earlier responses. Change as little wording as possible. Keep the answer highly relevant to the question, do not add extra stuff",
        max_tokens=150
    )
    return response.choices[0].text

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
    st.write(sorted_texts[:3])
    return list(zip(*sorted_texts[:3]))
    # df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    # sorted_by_similarity = df.sort_values("similarities", ascending=False).head(3)
    # if sorted_by_similarity.iloc[2,4] < 0.8:
    #     results = "Question is out of scope. Please try to rephrase it."
    # else:
    #     results = sorted_by_similarity['text'].values.tolist()
    #     response = craft_response(text_input, results)
    #     response = "Q:" + text_input + " A" + response

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
uploaded_file = st.file_uploader("Choose a file first")
st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
      st.write(uploaded_file.read())
    #   with open(f"data:application/pdf;base64/{uploaded_file.name}", "rb") as f:
    #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    #     num_pages = base64_pdf.numPages
    #     count = 0
    #     text = ""
    #     while count < num_pages:
    #       pageObj = base64_pdf.getPage(count)
    #       count +=1
    #       text += pageObj.extractText()
    #       texts = text_splitter.split_text(text)
    # else:
    #   texts = text_splitter.split_text(uploaded_file.read().decode("utf-8"))
    # text_vectors = []
    # for i in range(len(texts)):
    #   text_vectors.append(get_embedding(texts[i], engine="text-embedding-ada-002"))
    # text_input = st.text_input(
    #     "Ask a question 👇", # make this custom to the pdf
    #     label_visibility=st.session_state.visibility,
    #     disabled=st.session_state.disabled
    #     # placeholder=st.session_state.placeholder,
    # )
    # similar_terms = get_similar_terms(text_input, text_vectors, texts)
    # response = craft_response(text_input, similar_terms)
    # if text_input:
    #     st.write("Answer: " + response)


# # with open("foo.pkl", 'rb') as f: 
# #    new_docsearch = pickle.load(f)

# # docsearch = FAISS.from_texts(texts, new_docsearch)
# # print(docsearch)
# # query = "How much will sea level rise"
# # docs = docsearch.similarity_search(query)
# # # # print(docs[0].page_content)
# # # # import pinecone
# # # # pinecone.init(api_key="YOUR_API_KEY",
# # # #               environment="us-west1-gcp")

# # # # pinecone.create_index("example-index", dimension=1024)
# # chain = load_qa_chain(OpenAI(temperature=0, openai_api_key='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'), chain_type="stuff")
# # answer = chain.run(input_documents=docs, question=query)
# # print(query)
# # print(answer)