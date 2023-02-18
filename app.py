from typing import Dict
import pandas as pd
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from langchain.text_splitter import TokenTextSplitter
from openai.embeddings_utils import get_embedding, cosine_similarity
from operator import itemgetter
import openai
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
import os
from streamlit_chat import message

openai.api_key = st.secrets["api_key"]
os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

st.title("Ask PDF Anything")
@st.cache_resource()
def extract_text_from_pdfs(pdf_files):
    # Create an empty data frame
    df = pd.DataFrame(columns=["file", "text"])
    # Iterate over the PDF files
    for pdf_file in pdf_files:
        # Open the PDF file
        # with open(pdf_file.read(), "rb") as f:
        with BytesIO(pdf_file.read()) as f:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(f)
            # Get the number of pages in the PDF
            num_pages = len(pdf_reader.pages)
            # Initialize a string to store the text from the PDF
            text = ""
            # Iterate over all the pages
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract the text from the page
                page_text = page.extract_text()
                # Add the page text to the overall text
                text += page_text
            # Add the file name and the text to the data frame
            df = df.append({"file": pdf_file.name, "text": text}, ignore_index=True)
    # Return the data frame
    return df

text_splitter = TokenTextSplitter(chunk_size = 200, chunk_overlap = 40)

@st.cache_resource()
def get_context(text_input, text_vectors, texts):
    search_term_vector = get_embedding(text_input, engine="text-embedding-ada-002")
    similarities = []
    for i in range(len(text_vectors)):
        similarities.append(cosine_similarity(text_vectors[i], search_term_vector))
    df = pd.DataFrame({'text': texts, 'similarity': similarities})
    sorted_similarities = df.sort_values(by=['similarity'], ascending=False, ignore_index=True)
    st.write(sorted_similarities)
    return [str(sorted_similarities['text'][0]), str(sorted_similarities['text'][1]), str(sorted_similarities['text'][2])]

pdf_files = st.file_uploader(
    "Upload pdf files", type=["pdf"], accept_multiple_files=True
)

template = """You are a highly intelligent question answering bot. Given the following extracted parts of a long document create a response to the question. Stick entirely to the facts given within the context. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
CONTEXT: {context}
:"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0), 
    prompt=PROMPT, 
    verbose=True,
)

if pdf_files:
    with st.spinner("processing pdf..."):
        df = extract_text_from_pdfs(pdf_files)
    text_vectors = []
    texts = []
    for i in range(len(df)):
      texts.extend(text_splitter.split_text(df['text'][i]))
    for i in range(len(texts)):
      text_vectors.append(get_embedding(texts[i], engine="text-embedding-ada-002"))
    question = st.text_input("Enter your questions here...")
    if question:
        with st.spinner("Searching. Please hold..."):
            context = get_context(question, text_vectors, texts)
            response = chatgpt_chain.run({"context": context, "question": question})
            st.write(response)
            st.write(context)
    # if question:
    #     if 'generated' not in st.session_state:
    #       st.session_state['generated'] = []

    #     if 'past' not in st.session_state:
    #       st.session_state['past'] = []
    #     st.session_state.past.append(question)
    #     st.session_state.generated.append(response)
    #     if st.session_state['generated']:
    #       for i in range(len(st.session_state['generated'])-1, -1, -1):
    #           message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    #           message(st.session_state["generated"][i], key=str(i))