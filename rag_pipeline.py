from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

def build_rag_pipeline():
    # Step 1: Create sample documents
    documents = [
        "Cerence builds AI-powered voice assistants for cars.",
        "Retrieval-Augmented Generation improves accuracy by adding external context.",
        "Transformers are widely used in NLP tasks."
    ]

    # Step 2: Convert to embeddings
    embeddings = HuggingFaceEmbeddings()

    # Step 3: Store in vector database
    vectorstore = FAISS.from_texts(documents, embeddings)

    # Step 4: Create retriever
    retriever = vectorstore.as_retriever()

    # Step 5: Load LLM
    generator = pipeline("text-generation", model="gpt2")
    llm = HuggingFacePipeline(pipeline=generator)

    # Step 6: Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain


def ask_question(query):
    qa_chain = build_rag_pipeline()
    result = qa_chain.run(query)
    return result
