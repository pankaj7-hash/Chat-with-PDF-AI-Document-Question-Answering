import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain imports (0.3.x)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ---------- Setup ----------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INDEX_DIR = "faiss_index"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- Helpers ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs or []:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=EMBEDDINGS)
    vector_store.save_local(INDEX_DIR)

def build_rag_chain():
    """Replaces load_qa_chain with the new LCEL pipeline."""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question using ONLY the provided context.
If answer is not in context, reply:
"answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )

    # LCEL pipeline
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def answer_question(user_question: str):
    new_db = FAISS.load_local(
        INDEX_DIR,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question, k=4)

    # Combine retrieved docs
    context = "\n\n".join([d.page_content for d in docs])

    chain = build_rag_chain()

    response = chain.invoke(
        {"context": context, "question": user_question}
    )

    st.write("Reply:", response)

# ---------- UI ----------
def main():
    st.set_page_config(
        page_title="Chat PDF",
        page_icon="📄",
        menu_items={
            "About": "Chat with PDF — © 2025 Pankaj Mahure\n"
        },
    )

    st.header("Chat with PDF 💁")
    st.markdown(
        "Built by **Pankaj Mahure** · "
    )

    st.caption(
        f"Backend: **GROQ** | Embeddings: **all-MiniLM-L6-v2** | Index: **{INDEX_DIR}**"
    )

    user_question = st.text_input("Ask a question about your PDFs")

    if user_question:
        if not os.path.isdir(INDEX_DIR):
            st.error("❌ No index found. Upload PDFs and click 'Submit & Process' first.")
        else:
            try:
                answer_question(user_question)
            except Exception as e:
                st.error(f"Error answering: {e}")

    # Sidebar
    with st.sidebar:
        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDFs, then click 'Submit & Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing…"):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)

                    if os.path.isdir(INDEX_DIR):
                        shutil.rmtree(INDEX_DIR)

                    get_vector_store(chunks)
                    st.success("✅ Done! You can now ask questions.")
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")

if __name__ == "__main__":
    main()
