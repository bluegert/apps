# from flask import Flask, request, render_template
# import openai
# from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
# import numpy as np
# import os
# from supabase import create_client, Client
# import asyncio
import streamlit as st




# url: str = "https://dlorxtrnmfxnyttxbsnp.supabase.co"
# key: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsb3J4dHJubWZ4bnl0dHhic25wIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NzU3NzYxNDQsImV4cCI6MTk5MTM1MjE0NH0.jptsq-8M25X2nfVDDSmNicT7QUoVupQ3_QlM4ee-qCo'
# supabase: Client = create_client(url, key)

# app = Flask(__name__)

# openai.api_key = "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"

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

st.write(df)


# @app.route('/static/')
# def serve_static(filename):
#   return app.send_static_file(filename)

# @app.route('/')
# def search_form():
#   return render_template('search_form.html')


# def craft_response(query, msg):
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=f"You are a financial spokesman for Microsoft. A journalist just asked you this question: {query}. In the past you have said the following around this topic: {msg}. Craft an insightful response to the journalist based on the facts presented in your earlier responses. Change as little wording as possible. Keep the answer highly relevant to the question, do not add extra stuff",
#         max_tokens=150
#     )
#     return response.choices[0].text

# @app.route('/search')
# def search():
#     # Get the search query from the URL query string
#     query = request.args.get('query')

#     search_term_vector = get_embedding(query, engine="text-embedding-ada-002")
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
#     supabase.table("QnA").insert([{"Q": query, "A": response}]).execute() # make async
#     # Render the search results template, passing in the search query and results
#     return render_template('search_results.html', query=query, results=response)

# if __name__ == '__main__':
#   app.run()
