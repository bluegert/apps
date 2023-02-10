# from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import os
# from supabase import create_client, Client
# import asyncio
import streamlit as st
from io import StringIO


openai.api_key = st.secrets['api_key']

def craft_response(query, msg):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"You are a financial spokesman for Microsoft. A journalist just asked you this question: {query}. In the past you have said the following around this topic: {msg}. Craft an insightful response to the journalist based on the facts presented in your earlier responses. Change as little wording as possible. Keep the answer highly relevant to the question, do not add extra stuff",
        max_tokens=150
    )
    return response.choices[0].text

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

def get_similar_terms(text_input, df):
    search_term_vector = get_embedding(text_input, engine="text-embedding-ada-002")
    st.write(df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector)))

    # df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    # sorted_by_similarity = df.sort_values("similarities", ascending=False).head(3)
    # if sorted_by_similarity.iloc[2,4] < 0.8:
    #     results = "Question is out of scope. Please try to rephrase it."
    # else:
    #     results = sorted_by_similarity['text'].values.tolist()
    #     response = craft_response(text_input, results)
    #     response = "Q:" + text_input + " A" + response
    response="remove this"
    return response

uploaded_file = st.file_uploader("Choose a file first", type="pdf")
if uploaded_file is not None:
    pdf = StringIO(uploaded_file)
    st.write(pdf.read())


    text_input = st.text_input(
        "Ask a question 👇", # make this custom to the pdf
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled
        # placeholder=st.session_state.placeholder,
    )
    # response = get_similar_terms(text_input, df)
    # if text_input:
    #     st.write("Answer: " + response)

  
# # creating a pdf file object
# pdfFileObj = open('/Users/wesley/AI/chatgpt/data_collection/docs/IPCC_AR6_WGII_SummaryForPolicyMakers.pdf', 'rb')
  
# # creating a pdf reader object
  
# raw_text = ''
# for i, page in enumerate(reader.pages):
#     text = page.extract_text()
#     if text:
#         raw_text += text

# text_splitter = CharacterTextSplitter(        
#     separator = "\n",
#     chunk_size = 1000,
#     chunk_overlap  = 200,
#     length_function = len,
# )
# texts = text_splitter.split_text(raw_text)
# embeddings = OpenAIEmbeddings(openai_api_key='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')
# import pickle
# # with open("foo.pkl", 'wb') as f:
# #     pickle.dump(embeddings, f)

# with open("foo.pkl", 'rb') as f: 
#    new_docsearch = pickle.load(f)

# docsearch = FAISS.from_texts(texts, new_docsearch)
# print(docsearch)
# query = "How much will sea level rise"
# docs = docsearch.similarity_search(query)
# # # print(docs[0].page_content)
# # # import pinecone
# # # pinecone.init(api_key="YOUR_API_KEY",
# # #               environment="us-west1-gcp")

# # # pinecone.create_index("example-index", dimension=1024)
# chain = load_qa_chain(OpenAI(temperature=0, openai_api_key='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'), chain_type="stuff")
# answer = chain.run(input_documents=docs, question=query)
# print(query)
# print(answer)