📄 Chat with PDF using GROQ + LangChain + Streamlit

An AI-powered RAG (Retrieval-Augmented Generation) application that allows users to upload PDFs and ask questions about their content. The system extracts text, creates embeddings, stores them in a FAISS vector database, and uses GROQ LLM (Llama-3.1-8B) to generate accurate answers based only on the uploaded documents.

Built with Streamlit, LangChain, FAISS, HuggingFace Embeddings, and GROQ API.

🚀 Features

Upload multiple PDF documents

Automatic text extraction and chunking

Vector embeddings using all-MiniLM-L6-v2

Fast similarity search using FAISS

Accurate answers using GROQ Llama-3.1-8B

Clean and simple Streamlit UI

Fully local vector storage

Context-based answering (prevents hallucinations)

🧠 Architecture
User Upload PDF
      │
      ▼
Text Extraction (PyPDF2)
      │
      ▼
Text Chunking (LangChain)
      │
      ▼
Embeddings (Sentence Transformers)
      │
      ▼
Vector Storage (FAISS)
      │
      ▼
User Question
      │
      ▼
Similarity Search
      │
      ▼
Context + Prompt
      │
      ▼
GROQ LLM (Llama-3.1)
      │
      ▼
Answer in Streamlit UI
📂 Project Structure
chat-with-pdf/
│
├── app.py
├── requirements.txt
├── faiss_index/          # generated automatically
├── .env
└── README.md
⚙️ Installation
1. Clone repository
git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf
2. Create virtual environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
🔑 Setup GROQ API Key

Create .env file:

GROQ_API_KEY=your_api_key_here

Get API key from:

https://console.groq.com/keys

▶️ Run the app
streamlit run app.py
🖥️ Usage

Upload one PDF file

Click Submit & Process

Ask questions related to the PDF

Get instant AI-generated answers

Example:

How to prevent diabetes?
🧾 Requirements
streamlit==1.39.0
langchain==0.3.6
langchain-core==0.3.15
langchain-community==0.3.6
langchain-text-splitters==0.3.0
langchain-groq==0.2.0
groq==0.11.0
sentence-transformers==2.7.0
faiss-cpu==1.8.0
python-dotenv==1.0.1
PyPDF2==3.0.1
🧩 Tech Stack

Streamlit

LangChain (LCEL)

GROQ LLM (Llama-3.1)

HuggingFace Embeddings

FAISS Vector Database

PyPDF2

Python

📊 Example Screenshot

(<img width="1836" height="913" alt="Chat with pdf2" src="https://github.com/user-attachments/assets/29f7ede7-19ed-42ae-ae69-5c6315aba20b" />

)

/screenshots/app.png
🎯 Use Cases

Medical document QA

Research paper assistant

Policy document search

Education material chatbot

Legal document assistant

🔒 Safety

The model answers only from provided context.
If answer is not found, it responds:

answer is not available in the context
👨‍💻 Author

Dr. Pankaj Mahure

Public Health Professional | AI Developer I Data Scientist

⭐ If you like this project

Please star the repository.

⭐ Star | Fork | Contribute
