# UPLOAD PDF file to and make chatbot around it that can discuss the contents of the pdf


import os
import streamlit as st

from apikey import apikey

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

os.environ['OPENAI_API_KEY'] = apikey

source_pdf = st.file_uploader("Upload a pdf", accept_multiple_files=False)

if source_pdf:
    reader = PdfReader(source_pdf)
    print(reader)

    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    print(raw_text)

    #split the pdf raw text in chunks spo it can be used by the model
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(),chain_type="stuff")

    query = st.text_input("Fill out what you want to know about this article")

    if query:
        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)

        st.write(answer)