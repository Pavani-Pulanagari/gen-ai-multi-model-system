# gen-ai-multi-model-system

## Overview
This project implements an end-to-end generative AI pipeline integrating multiple transformer-based models for text generation, language translation, and question answering.

The system leverages Hugging Face Transformers and LangChain to build a modular architecture capable of handling different NLP tasks within a unified workflow.

---

## Architecture

User Input → Task Router → Model Layer → Output

- Text Generation → GPT-2 (Hugging Face Transformers)
- Translation → MarianMT Model
- Question Answering → Retrieval-Augmented Generation (RAG)

The system uses a centralized pipeline to route tasks dynamically based on user input.

---

## Features

- Multi-model AI pipeline
- Modular architecture for scalability
- Retrieval-Augmented Generation (RAG) for improved contextual responses
- Secure API key handling using environment variables
- Support for multiple NLP tasks in a single system

---

## RAG Implementation

The RAG pipeline enhances response accuracy by incorporating external context:

1. Input documents are converted into vector embeddings using Hugging Face embeddings
2. Embeddings are stored in a FAISS vector database
3. Relevant documents are retrieved based on user query similarity
4. Retrieved context is passed to the language model for response generation

This approach reduces hallucination and improves contextual relevance compared to standalone LLM outputs.

---

## Optimization

- Implemented FP16 (half precision) to reduce memory usage and improve inference speed
- Designed modular components to allow batch processing and scalability

---

## Challenges

- Integrating multiple models into a unified pipeline
- Managing token limits in transformer-based models
- Designing effective prompts for consistent output quality
- Handling computational constraints in cloud environments

---

## Future Improvements

- Optimize latency for real-time applications
- Deploy models on edge devices
- Extend system to support voice-based AI interactions
- Improve retrieval mechanisms for domain-specific use cases

---

## Tech Stack

- Python
- Hugging Face Transformers
- LangChain
- FAISS (Vector Database)
- PyTorch

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

3. Choose a task:
- generate
- translate
- qa

## Summary

This project demonstrates the design of a scalable and modular AI system integrating multiple models and retrieval-based techniques. It highlights practical implementation of modern NLP workflows, including RAG, model optimization, and pipeline orchestration.
