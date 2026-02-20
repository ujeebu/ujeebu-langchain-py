#!/usr/bin/env python3
"""
Example: Using Ujeebu Document Loader with LangChain

This example demonstrates how to use UjeebuLoader to:
- Load articles into LangChain Document objects
- Use with vector stores for semantic search
- Build Q&A systems over web content

Prerequisites:
    - pip install langchain-ujeebu langchain-openai faiss-cpu
    - Set UJEEBU_API_KEY environment variable
    - Set OPENAI_API_KEY environment variable (for vector store examples)

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
from langchain_ujeebu import UjeebuLoader

# Ensure API key is set
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    print("Get your API key at: https://ujeebu.com/signup")
    exit(1)


def load_single_article():
    """Load a single article as a LangChain Document."""
    print("\n" + "=" * 60)
    print("Example 1: Load Single Article")
    print("=" * 60)

    loader = UjeebuLoader(
        urls=["https://www.bbc.com/news/technology-67881954"],
        extract_text=True,
        extract_author=True,
        extract_pub_date=True,
    )

    documents = loader.load()

    print(f"\nLoaded {len(documents)} document(s)")

    if documents:
        doc = documents[0]
        print("\n📄 Document Details:")
        print(f"   Title: {doc.metadata.get('title', 'N/A')}")
        print(f"   Author: {doc.metadata.get('author', 'N/A')}")
        print(f"   Published: {doc.metadata.get('pub_date', 'N/A')}")
        print(f"   Source: {doc.metadata.get('source', 'N/A')}")
        print(f"\n   Content preview:")
        print(f"   {doc.page_content[:500]}...")

    return documents


def load_multiple_articles():
    """Load multiple articles at once."""
    print("\n" + "=" * 60)
    print("Example 2: Load Multiple Articles")
    print("=" * 60)

    urls = [
        "https://www.bbc.com/news/technology-67881954",
        "https://www.theverge.com/2024/1/16/24040562/google-layoffs-2024",
        "https://techcrunch.com/2024/01/15/openai-gpt-store/",
    ]

    loader = UjeebuLoader(
        urls=urls, extract_text=True, quick_mode=True  # Faster extraction
    )

    documents = loader.load()

    print(f"\nLoaded {len(documents)} documents from {len(urls)} URLs")

    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Unknown")[:50]
        source = doc.metadata.get("site_name", "Unknown")
        print(f"\n   {i}. {title}...")
        print(f"      Source: {source}")
        print(f"      Content length: {len(doc.page_content)} chars")

    return documents


def load_with_images():
    """Load articles including image metadata."""
    print("\n" + "=" * 60)
    print("Example 3: Load Articles with Images")
    print("=" * 60)

    loader = UjeebuLoader(
        urls=["https://www.bbc.com/news/technology-67881954"],
        extract_text=True,
        extract_images=True,
    )

    documents = loader.load()

    if documents:
        doc = documents[0]
        print(f"\n📷 Article: {doc.metadata.get('title', 'N/A')}")

        # Main article image
        if doc.metadata.get("image"):
            print(f"   Main image: {doc.metadata['image'][:80]}...")

        # All images
        images = doc.metadata.get("images", [])
        if images:
            print(f"   Total images: {len(images)}")
            for img in images[:3]:
                print(f"   - {img[:70]}...")

    return documents


def use_with_text_splitter():
    """Split loaded documents for vector storage."""
    print("\n" + "=" * 60)
    print("Example 4: Split Documents for Vector Store")
    print("=" * 60)

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Load articles
    loader = UjeebuLoader(
        urls=["https://www.bbc.com/news/technology-67881954"],
        extract_text=True,
    )
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\n📄 Original documents: {len(documents)}")
    print(f"📑 After splitting: {len(chunks)} chunks")

    if chunks:
        print(f"\n   Sample chunk (first 300 chars):")
        print(f"   {chunks[0].page_content[:300]}...")

    return chunks


def use_with_vector_store():
    """Create a vector store from loaded articles."""
    print("\n" + "=" * 60)
    print("Example 5: Create Vector Store from Articles")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return None

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    # Load multiple articles
    loader = UjeebuLoader(
        urls=[
            "https://www.bbc.com/news/technology-67881954",
            "https://techcrunch.com/2024/01/15/openai-gpt-store/",
        ],
        quick_mode=True,
    )
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"\n🔍 Vector store created with {len(chunks)} chunks")

    # Test similarity search
    query = "What are the latest AI developments?"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n   Query: '{query}'")
    print(f"   Found {len(results)} relevant chunks:")

    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Source: {result.metadata.get('title', 'Unknown')[:40]}...")
        print(f"      {result.page_content[:200]}...")

    return vectorstore


def build_qa_chain():
    """Build a Q&A chain over loaded articles."""
    print("\n" + "=" * 60)
    print("Example 6: Build Q&A Chain")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return None

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA

    # Load and process documents
    loader = UjeebuLoader(
        urls=[
            "https://www.bbc.com/news/technology-67881954",
            "https://techcrunch.com/2024/01/15/openai-gpt-store/",
        ],
        quick_mode=True,
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create Q&A chain
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

    # Ask questions
    questions = [
        "What is the main topic of these articles?",
        "Are there any mentions of AI companies?",
    ]

    print("\n🤖 Q&A Chain Results:")

    for question in questions:
        result = qa_chain.invoke({"query": question})
        print(f"\n   Q: {question}")
        print(f"   A: {result['result'][:300]}...")

    return qa_chain


if __name__ == "__main__":
    print("=" * 60)
    print("Ujeebu Document Loader Examples")
    print("=" * 60)

    # Run examples
    load_single_article()
    load_multiple_articles()
    load_with_images()
    use_with_text_splitter()

    # Vector store examples (require OpenAI API key)
    use_with_vector_store()
    build_qa_chain()

    print("\n" + "=" * 60)
    print("Document Loader Examples completed!")
    print("=" * 60)
