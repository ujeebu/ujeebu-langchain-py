#!/usr/bin/env python3
"""
Example: Building a RAG System with Ujeebu and LangChain

This example demonstrates how to build a Retrieval-Augmented Generation (RAG)
system using Ujeebu to extract web content and LangChain for the RAG pipeline.

Prerequisites:
    - pip install langchain-ujeebu langchain-openai faiss-cpu
    - Set UJEEBU_API_KEY environment variable
    - Set OPENAI_API_KEY environment variable

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
from langchain_ujeebu import UjeebuLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Check for required API keys
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    exit(1)

if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)


def build_rag_from_articles():
    """Build a RAG system from multiple articles."""
    print("\n" + "=" * 60)
    print("Building RAG System from Web Articles")
    print("=" * 60)

    # Step 1: Load articles using Ujeebu
    print("\n📰 Step 1: Loading articles with Ujeebu...")

    article_urls = [
        "https://ujeebu.com/blog/enhancing-lead-generation-with-web-data-scraping-and-content-extraction-2/",
        "https://ujeebu.com/blog/how-to-scrape-google-search-results/",
        "https://en.wikipedia.org/wiki/Web_scraping",
    ]

    loader = UjeebuLoader(
        urls=article_urls,
        extract_text=True,
        extract_author=True,
        extract_pub_date=True,
        quick_mode=True,  # Faster extraction
    )

    documents = loader.load()
    print(f"   Loaded {len(documents)} articles")

    for doc in documents:
        title = doc.metadata.get("title", "Unknown")[:50]
        print(f"   - {title}...")

    # Step 2: Split documents into chunks
    print("\n📄 Step 2: Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # Step 3: Create embeddings and vector store
    print("\n🔢 Step 3: Creating embeddings and vector store...")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"   Vector store created with {len(chunks)} vectors")

    # Step 4: Create RAG chain
    print("\n🔗 Step 4: Creating RAG chain...")

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # Custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always cite the source article when providing information.

    Context:
    {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    print("   RAG chain created successfully!")

    # Step 5: Query the RAG system
    print("\n❓ Step 5: Querying the RAG system...")

    questions = [
        "What are the main techniques for web scraping?",
        "How can web data be used for lead generation?",
        "What tools are recommended for scraping Google?",
    ]

    for question in questions:
        print(f"\n   Q: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"   A: {result['result'][:500]}...")

        # Show sources
        sources = [
            doc.metadata.get("title", "Unknown")[:40]
            for doc in result["source_documents"]
        ]
        print(f"   Sources: {', '.join(set(sources))}")

    return vectorstore, qa_chain


def rag_with_topic_focus():
    """Build a topic-focused RAG system."""
    print("\n" + "=" * 60)
    print("Topic-Focused RAG: AI News Aggregator")
    print("=" * 60)

    # Load AI-focused articles
    ai_news_urls = [
        "https://ujeebu.com/blog/enhancing-lead-generation-with-web-data-scraping-and-content-extraction-2/",
        "https://ujeebu.com/blog/how-to-scrape-google-search-results/",
    ]

    print("\n📰 Loading AI news articles...")
    loader = UjeebuLoader(urls=ai_news_urls, quick_mode=True)
    documents = loader.load()

    # Create specialized chunking for news
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for news
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks from {len(documents)} articles")

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create specialized AI news assistant
    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

    ai_news_prompt = """You are an AI news analyst. Based on the following news excerpts,
    provide a concise and informative answer. Focus on facts and recent developments.

    News Context:
    {context}

    User Question: {question}

    Analysis:"""

    prompt = PromptTemplate(
        template=ai_news_prompt, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": prompt},
    )

    # Test queries
    print("\n❓ Querying AI news RAG system...")

    result = qa_chain.invoke({"query": "What are the key web scraping techniques discussed?"})
    print(f"\n   Summary: {result['result']}")


def incremental_rag():
    """Demonstrate incremental document addition to RAG."""
    print("\n" + "=" * 60)
    print("Incremental RAG: Adding Documents Over Time")
    print("=" * 60)

    # Initial document load
    print("\n📰 Initial load: Loading first batch of articles...")

    initial_urls = [
        "https://ujeebu.com/blog/enhancing-lead-generation-with-web-data-scraping-and-content-extraction-2/",
    ]

    loader = UjeebuLoader(urls=initial_urls, quick_mode=True)
    initial_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    initial_chunks = text_splitter.split_documents(initial_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(initial_chunks, embeddings)
    print(f"   Initial vector store: {len(initial_chunks)} chunks")

    # Add more documents later
    print("\n📰 Adding new articles to existing RAG...")

    new_urls = [
        "https://ujeebu.com/blog/how-to-scrape-google-search-results/",
    ]

    new_loader = UjeebuLoader(urls=new_urls, quick_mode=True)
    new_docs = new_loader.load()
    new_chunks = text_splitter.split_documents(new_docs)

    # Add to existing vector store
    vectorstore.add_documents(new_chunks)
    print(f"   Added {len(new_chunks)} new chunks")
    print(f"   Total vectors now: {vectorstore.index.ntotal}")

    # Query the updated system
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )

    result = qa_chain.invoke({"query": "What topics are covered in these articles?"})
    print(f"\n   Answer: {result['result']}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG System Examples with Ujeebu + LangChain")
    print("=" * 60)

    # Run examples
    build_rag_from_articles()
    rag_with_topic_focus()
    incremental_rag()

    print("\n" + "=" * 60)
    print("RAG Examples completed!")
    print("=" * 60)
