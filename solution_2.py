import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import re


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
    return text


def sanitize_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove common irrelevant patterns (customize as needed)
    patterns_to_remove = [
        r'DOI:\s*\d+.\d+/\d+',  # DOI
        r'CITATIONS\s*\d+',  # Citations count
        r'READS\s*\d+',  # Reads count
        # Add more patterns here
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)

    return text


def preprocess_text(text):
    # Replace certain patterns with newlines or another separator
    text = text.replace(":", "\n")  # Example: replace colons with newlines
    # Add more replacements as needed based on your analysis of the PDFs' structure
    return text


def get_text_chunks(text):
    separator = "\n\n"

    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    print(f"Total chunks before refinement: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} size: {len(chunk)}")

    refined_chunks = []
    for chunk in chunks:
        # Further split chunks if they're too large
        while len(chunk) > 1000:
            # Find the last occurrence of the separator before the 1000th character
            split_pos = chunk.rfind(separator, 0, 1000)
            if split_pos == -1:  # If no separator is found, forcibly split at 1000
                split_pos = 1000
            part, chunk = chunk[:split_pos], chunk[split_pos:]
            refined_chunks.append(part)
        if chunk:  # Add the last part if it's not empty
            refined_chunks.append(chunk)

    print(f"Total chunks after refinement: {len(refined_chunks)}")
    for i, chunk in enumerate(refined_chunks):
        print(f"Refined chunk {i} size: {len(chunk)}")

    return refined_chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_contextual_compression_retriever(vectorstore, llm):
    # Create a compressor - you can use LLMChainExtractor or any other compressor you prefer
    compressor = LLMChainExtractor.from_llm(llm)

    # Create the Contextual Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever()
    )
    return compression_retriever


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    compression_retriever = get_contextual_compression_retriever(vectorstore, llm)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)

                preprocessed_text = preprocess_text(raw_text)

                sanitized_text = sanitize_text(preprocessed_text)

                # get the text chunks
                text_chunks = get_text_chunks(sanitized_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()