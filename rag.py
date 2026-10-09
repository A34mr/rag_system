# rag.py
"""
Modern RAG core. Uses LangChain 1.x-style modular packages.
No langchain-community, no langchain.chains.
"""
import os

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# --- config ---------------------------------------------------------------
DATA_DIR    = "data"
DB_DIR      = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "llama3.1"
CHUNK_SIZE  = 800
CHUNK_OVER  = 150
TOP_K       = 4
# -------------------------------------------------------------------------


def load_documents():
    """Load .pdf, .txt, .md files from DATA_DIR without langchain-community."""
    from pypdf import PdfReader

    docs = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".pdf":
                try:
                    reader = PdfReader(path)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": path, "page": i},
                            ))
                except Exception as e:
                    print(f"[warn] failed to read {path}: {e}")

            elif ext in {".txt", ".md"}:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        docs.append(Document(
                            page_content=f.read(),
                            metadata={"source": path},
                        ))
                except Exception as e:
                    print(f"[warn] failed to read {path}: {e}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVER,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},       # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore():
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .pdf/.txt/.md files found under {DATA_DIR}/")

    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} pages/docs → {len(chunks)} chunks")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
    )
    print(f"Saved vector store to {DB_DIR}/")
    return vectordb


def load_vectorstore():
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
    )


PROMPT = PromptTemplate(
    template="""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source','?')}"
        + (f" p.{d.metadata['page']}" if 'page' in d.metadata else "")
        + f"]\n{d.page_content}"
        for d in docs
    )


def build_qa_chain(vectordb):
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.2)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


if __name__ == "__main__":
    # running `python rag.py` directly just builds the index
    build_vectorstore()