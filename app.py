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
@st.cache(allow_output_mutation=True)
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

@st.cache(allow_output_mutation=True)
def get_context(text_input, text_vectors, texts):
    search_term_vector = get_embedding(text_input, engine="text-embedding-ada-002")
    similarities = []
    for i in range(len(text_vectors)):
        similarities.append(cosine_similarity(text_vectors[i], search_term_vector))
    df = pd.DataFrame({'text': texts, 'similarity': similarities})
    sorted_similarities = df.sort_values(by=['similarity'], ascending=False)
    return str(sorted_similarities['text'][0]) + str(sorted_similarities['text'][1]) + str(sorted_similarities['text'][2]) 

pdf_files = st.file_uploader(
    "Upload pdf files", type=["pdf"], accept_multiple_files=True
)

template = """Given the following extracted parts of a long document and a question. 
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
    texts = text_splitter.split_text(df['text'][0])
    for i in range(len(texts)):
      text_vectors.append(get_embedding(texts[i], engine="text-embedding-ada-002"))
    question = st.text_input("Enter your questions here...")
    if question:
        with st.spinner("Searching. Please hold..."):
            context = get_context(question, text_vectors, texts)
            response = chatgpt_chain.run({"context": context, "question": question})
            st.write(response)
        # label_visibility=st.session_state.visibility,
        # disabled=st.session_state.disabled
        # placeholder=st.session_state.placeholder,
    # )

    if question:
      if 'generated' not in st.session_state:
            st.session_state['generated'] = []

      if 'past' not in st.session_state:
          st.session_state['past'] = []
      # if already text generated, build on that
      if st.session_state['generated']:
        st.write(st.session_state['generated'])
        st.session_state.past.append(question)
      st.session_state.generated.append(response)
      if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

# # # pdf support
# # # https://discuss.streamlit.io/t/how-to-display-pdf-files-in-streamlit/1806/2
# # # prompts building on each other
# # # summarizer
# # # better prompts (see youtube gpt)
# # # multiple files upload
# # # making it look nicer
# # # data privacy
# # # some metric of performance. Can I give the model feedback? Should I?
# # organization key or private key
# # pinecone
# # the ideal dataset over time becomes more and more specific to the user and in Q&A format
# # check if embedding already exists
    # text_input = st.text_input(
        # "Ask a question 👇", # make this custom to the pdf
        # label_visibility=st.session_state.visibility,
        # disabled=st.session_state.disabled
        # placeholder=st.session_state.placeholder,
    # )

#     if text_input:
#       if 'generated' not in st.session_state:
#             st.session_state['generated'] = []

#       if 'past' not in st.session_state:
#           st.session_state['past'] = []
#       similar_terms = get_similar_terms(text_input, text_vectors, texts)
#       user_input_embedding_prompt = 'Using this context: "'+str(similar_terms[0])+'", answer the following question changing as little wording as possible of the context. \n'+ text_input
#       st.write(user_input_embedding_prompt)
# #       response = generate_response(user_input_embedding_prompt)
# #       # if already text generated, build on that
# #       if st.session_state['generated']:
# #         st.write(st.session_state['generated'])
# #       st.session_state.past.append(text_input)
# #       st.session_state.generated.append(response)
# #       if st.session_state['generated']:
# #         for i in range(len(st.session_state['generated'])-1):
# #             message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
# #             message(st.session_state["generated"][i], key=str(i))

# # # pdf support
# # # https://discuss.streamlit.io/t/how-to-display-pdf-files-in-streamlit/1806/2
# # # prompts building on each other
# # # summarizer
# # # better prompts (see youtube gpt)
# # # multiple files upload
# # # making it look nicer
# # # data privacy
# # # some metric of performance. Can I give the model feedback? Should I?
# # organization key or private key
# # pinecone
# # the ideal dataset over time becomes more and more specific to the user and in Q&A format
# # check if embedding already exists