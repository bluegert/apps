from typing import Dict
import pandas as pd
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, BertForQuestionAnswering, pipeline
from io import BytesIO

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

@st.cache(allow_output_mutation=True)
def get_context(df, question):
    pass

def answer_question(pipeline, question: str, context: str) -> Dict:
    input = {"question": question, "context": context}
    return pipeline(input)

pdf_files = st.file_uploader(
    "Upload pdf files", type=["pdf"], accept_multiple_files=True
)

if pdf_files:
    with st.spinner("processing pdf..."):
        df = extract_text_from_pdfs(pdf_files)
    question = st.text_input("Enter your questions here...")
    st.write(df['text'])
    # if question != "":
    #     with st.spinner("Searching. Please hold..."):
    #         context = get_context(df, question)
    #         st.write(context)
            # answer = answer_question(question, context)
            # st.write(answer)
    #     del qa_pipeline
    #     del context