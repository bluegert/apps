# # from flask import Flask, request, render_template
# import openai
# from openai.embeddings_utils import get_embedding, cosine_similarity
# import pandas as pd
# import numpy as np
# import os
# # from supabase import create_client, Client
# # import asyncio
# import streamlit as st
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS


# openai.api_key = st.secrets['api_key']

# def craft_response(query, msg):
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=f"You are a financial spokesman for Microsoft. A journalist just asked you this question: {query}. In the past you have said the following around this topic: {msg}. Craft an insightful response to the journalist based on the facts presented in your earlier responses. Change as little wording as possible. Keep the answer highly relevant to the question, do not add extra stuff",
#         max_tokens=150
#     )
#     return response.choices[0].text

# # Store the initial value of widgets in session state
# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False

# # def get_similar_terms(text_input, df):
# #     search_term_vector = get_embedding(text_input, engine="text-embedding-ada-002")
# #     st.write(df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector)))

# #     # df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
# #     # sorted_by_similarity = df.sort_values("similarities", ascending=False).head(3)
# #     # if sorted_by_similarity.iloc[2,4] < 0.8:
# #     #     results = "Question is out of scope. Please try to rephrase it."
# #     # else:
# #     #     results = sorted_by_similarity['text'].values.tolist()
# #     #     response = craft_response(text_input, results)
# #     #     response = "Q:" + text_input + " A" + response
# #     response="remove this"
# #     return response

# text_splitter = CharacterTextSplitter(        
#     separator = "\n",
#     chunk_size = 1000,
#     chunk_overlap  = 200,
#     length_function = len,
# )

# uploaded_file = st.file_uploader("Choose a file first")
# if uploaded_file is not None:
#     texts = text_splitter.split_text(uploaded_file.read().decode("utf-8"))
#     embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
#     docsearch = FAISS.from_texts(texts, embeddings)
#     text_input = st.text_input(
#         "Ask a question 👇", # make this custom to the pdf
#         label_visibility=st.session_state.visibility,
#         disabled=st.session_state.disabled
#         # placeholder=st.session_state.placeholder,
#     )
#     query = "How do I send money"
#     response = docsearch.similarity_search(query)
#     st.write(response)
#     craft_response(query, response)
#     if text_input:
#         st.write("Answer: " + response)


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

import pandas as pd
import numpy as np
import streamlit as st
import whisper
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import os

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''
# Sidebar
with st.sidebar:
    user_secret = st.text_input(label = ":blue[OpenAI API key]",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    youtube_link = st.text_input(label = ":red[Youtube link]",
                                placeholder = "")
    if youtube_link and user_secret:
        youtube_video = YouTube(youtube_link)
        streams = youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        if st.button("Start Analysis"):
            if os.path.exists("word_embeddings.csv"):
                os.remove("word_embeddings.csv")
                
            with st.spinner('Running process...'):
                # Get the video mp4
                mp4_video = stream.download(filename='youtube_video.mp4')
                audio_file = open(mp4_video, 'rb')
                st.write(youtube_video.title)
                st.video(youtube_link) 

                # Whisper
                output = model.transcribe("youtube_video.mp4")
                
                # Transcription
                transcription = {
                    "title": youtube_video.title.strip(),
                    "transcription": output['text']
                }
                data_transcription.append(transcription)
                pd.DataFrame(data_transcription).to_csv('transcription.csv') 

                # Embeddings
                segments = output['segments']
                for segment in segments:
                    openai.api_key = user_secret
                    response = openai.Embedding.create(
                        input= segment["text"].strip(),
                        model="text-embedding-ada-002"
                    )
                    embeddings = response['data'][0]['embedding']
                    meta = {
                        "text": segment["text"].strip(),
                        "start": segment['start'],
                        "end": segment['end'],
                        "embedding": embeddings
                    }
                    data.append(meta)
                pd.DataFrame(data).to_csv('word_embeddings.csv') 
                st.success('Analysis completed')

st.title("Youtube GPT 🤖 ")
tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Transcription", "Embedding", "Chat with the Video"])
with tab1:
    st.markdown('Read the article to know how it works: [Medium Article]("https://medium.com/@dan.avila7/youtube-gpt-start-a-chat-with-a-video-efe92a499e60")')
    st.markdown("### How does it work?")
    st.write("Youtube GPT was written with the following tools:")
    st.markdown("#### Code GPT")
    st.write("All code was written with the help of Code GPT. Visit [codegpt.co]('https://codegpt.co') to get the extension.")
    st.markdown("#### Streamlit")
    st.write("The design was written with [Streamlit]('https://streamlit.io/').")
    st.markdown("#### Whisper")
    st.write("Video transcription is done by [OpenAI Whisper]('https://openai.com/blog/whisper/').")
    st.markdown("#### Embedding")
    st.write('[Embedding]("https://platform.openai.com/docs/guides/embeddings") is done via the OpenAI API with "text-embedding-ada-002"')
    st.markdown("#### GPT-3")
    st.write('The chat uses the OpenAI API with the [GPT-3]("https://platform.openai.com/docs/models/gpt-3") model "text-davinci-003""')
    st.markdown("""---""")
    st.write('Repo: [Github](https://github.com/davila7/youtube-gpt)')

with tab2: 
    st.header("Transcription:")
    if(os.path.exists("youtube_video.mp4")):
        audio_file = open('youtube_video.mp4', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
    if os.path.exists("transcription.csv"):
        df = pd.read_csv('transcription.csv')
        st.write(df)
with tab3:
    st.header("Embedding:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv('word_embeddings.csv')
        st.write(df)
with tab4:
    st.header("Ask me something about the video:")
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        input_text = st.text_input("You: ","", key="input")
        return input_text

    user_input = get_text()

    def get_embedding_text(api_key, prompt):
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input= prompt.strip(),
            model="text-embedding-ada-002"
        )
        q_embedding = response['data'][0]['embedding']
        df=pd.read_csv('word_embeddings.csv', index_col=0)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)

        df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
        returns = []
        
        # Sort by distance with 2 hints
        for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)

    def generate_response(api_key, prompt):
        one_shot_prompt = '''I am YoutubeGPT, a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
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
            temperature=0.5,
        )
        message = completions.choices[0].text
        return message

    if user_input:
        text_embedding = get_embedding_text(user_secret, user_input)
        title = pd.read_csv('transcription.csv')['title']
        string_title = "\n\n###\n\n".join(title)
        user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
        # st.write(user_input_embedding)
        output = generate_response(user_secret, user_input_embedding)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')




