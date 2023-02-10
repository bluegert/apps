# from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import os
# from supabase import create_client, Client
# import asyncio
import streamlit as st

openai.api_key = "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"

def craft_response(query, msg):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"You are a financial spokesman for Microsoft. A journalist just asked you this question: {query}. In the past you have said the following around this topic: {msg}. Craft an insightful response to the journalist based on the facts presented in your earlier responses. Change as little wording as possible. Keep the answer highly relevant to the question, do not add extra stuff",
        max_tokens=150
    )
    return response.choices[0].text

import streamlit as st

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col2 = st.columns(1)

uploaded_file = st.file_uploader("Choose a file", type="csv")


def get_similar_terms(text_input):
    search_term_vector = get_embedding(text_input, engine="text-embedding-ada-002")
    # df = dict(supabase.table("Vec").select("*").execute())
    df = pd.DataFrame.from_dict(df['data'])
    df['similarity'] = df['vec'].apply(lambda x: cosine_similarity(x, search_term_vector))
    sorted_by_similarity = df.sort_values("similarity", ascending=False).head(3)
    msg = ''
    return msg

with col2:
    text_input = st.text_input(
        "Ask a question about Microsoft's latest shareholder meeting 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )
    msg = 'We are profitable'
    # msg = get_similar_terms()

    if text_input:
        craft_response(text_input, msg)
        st.write("Answer: ", text_input)


#     df = dict(supabase.table("Vec").select("*").execute())
#     df = pd.DataFrame.from_dict(df['data'])
#     # df = pd.read_csv('/Users/wesley/AI/chatgpt/earnings_embeddings.csv')
#     df['embedding'] = df['embedding'].apply(eval).apply(np.array)
#     df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
#     sorted_by_similarity = df.sort_values("similarities", ascending=False).head(3)
#     # print(sorted_by_similarity.iloc[2,3])
#     print(sorted_by_similarity.text)
#     if sorted_by_similarity.iloc[2,3] < 0.8:
#       response = "Question is out of scope. Please try to rephrase it."
#     else:
#       results = sorted_by_similarity['text'].values.tolist()
#       response = craft_response(query, results)
#       response = "Q:" + query + " A" + response
#     # Render the search results template, passing in the search query and results
#     return render_template('search_results.html', query=query, results=response)
