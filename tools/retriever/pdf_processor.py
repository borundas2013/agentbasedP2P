from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initialize PDF processor with configurable chunk parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the text splitter with optimal settings for PDFs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],  # Better separators for large PDFs
            is_separator_regex=False
        )

        
        # Initialize embedding model with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.llm = ChatOpenAI(model='ft:gpt-4o-mini-2024-07-18:personal::BKodpSOI', api_key=api_key, temperature=0)
    
    def load_pdf(self, file_path: str) -> List:
    
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            docs= docs[50:-50]
            print(f"Successfully loaded {len(docs)} pages from {file_path}")
            return docs
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")
    
    def split_documents(self, documents: List) -> List:
    
        try:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            raise Exception(f"Error splitting documents: {str(e)}")
    
    def create_embeddings(self, documents: List) -> List:
        """
        Create embeddings for the given documents.
        
        Args:
            documents: List of document objects
            
        Returns:
            List of documents with embeddings
        """
        try:
            print(f"Creating embeddings for {len(documents)} documents...")
            # The embeddings will be created when we add documents to ChromaDB
            return documents
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def store_in_chromadb(self, documents: List, collection_name: str = "pdf_documents", 
                         persist_directory: str = "./chroma_db") -> Chroma:
        
        try:
            print(f"Storing {len(documents)} documents in ChromaDB collection: {collection_name}")
            
            # Create ChromaDB vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            
            # Note: persist() is no longer needed in Chroma 0.4.x - docs are auto-persisted
            print(f"Successfully stored documents in ChromaDB at {persist_directory}")
            
            return vectorstore
            
        except Exception as e:
            raise Exception(f"Error storing in ChromaDB: {str(e)}")
    
    def load_from_chromadb(self, collection_name: str = "pdf_documents", 
                          persist_directory: str = "./chroma_db") -> Chroma:
        """
        Load existing ChromaDB collection.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory where ChromaDB data is persisted
            
        Returns:
            ChromaDB vector store instance
        """
        try:
            print(f"Loading ChromaDB collection: {collection_name}")
            
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            
            # Get collection info
            collection = vectorstore._collection
            count = collection.count()
            print(f"Loaded collection with {count} documents")
            
            return vectorstore
            
        except Exception as e:
            raise Exception(f"Error loading from ChromaDB: {str(e)}")
    
    def search_similar(self, query: str, vectorstore: Chroma, k: int = 5) -> List:
        """
        Search for similar documents in ChromaDB.
        
        Args:
            query: Search query
            vectorstore: ChromaDB vector store instance
            k: Number of similar documents to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            print(f"Searching for documents similar to: '{query}'")
            
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            print(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            raise Exception(f"Error searching in ChromaDB: {str(e)}")
    
    def max_marginal_relevance_search(self, query: str, vectorstore: Chroma, k: int = 5, 
                                    fetch_k: int = 20, lambda_mult: float = 0.5) -> List:
        """
        Search for documents using Max Marginal Relevance to ensure diversity.
        
        Args:
            query: Search query
            vectorstore: ChromaDB vector store instance
            k: Number of documents to return
            fetch_k: Number of documents to fetch before filtering
            lambda_mult: Diversity parameter (0 = max diversity, 1 = max relevance)
            
        Returns:
            List of diverse documents
        """
        try:
            print(f"Performing MMR search for: '{query}'")
            
            results = vectorstore.max_marginal_relevance_search(
                query, 
                k=k, 
                fetch_k=fetch_k, 
                lambda_mult=lambda_mult
            )
            
            print(f"Found {len(results)} diverse documents")
            return results
            
        except Exception as e:
            raise Exception(f"Error in MMR search: {str(e)}")
    
    def process_pdf(self, file_path: str) -> List:
       
        # Load the PDF
        docs = self.load_pdf(file_path)
        
        
        # Split the documents
        split_docs = self.split_documents(docs)
        
        return split_docs
    
    def process_and_store_pdf(self, file_path: str, collection_name: str = "pdf_documents",
                            persist_directory: str = "./chroma_db") -> Chroma:
        """
        Complete pipeline: load PDF, split, create embeddings, and store in ChromaDB.
        
        Args:
            file_path: Path to the PDF file
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            
        Returns:
            ChromaDB vector store instance
        """
        # Process the PDF
        processed_docs = self.process_pdf(file_path)
        
        # Create embeddings and store in ChromaDB
        vectorstore = self.store_in_chromadb(processed_docs, collection_name, persist_directory)
        
        return vectorstore
    
    def create_optimized_retriever(self, vectorstore: Chroma, k: int = 8, 
                                 similarity_threshold: float = 0.7) -> any:
        """
        Create an optimized retriever with better parameters.
        
        Args:
            vectorstore: ChromaDB vector store instance
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            Optimized retriever
        """
        try:
            # Create retriever with better parameters
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": similarity_threshold
                }
            )
            
            return retriever
            
        except Exception as e:
            raise Exception(f"Error creating optimized retriever: {str(e)}")

    def get_document_info(self, documents: List) -> dict:
    
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents) if documents else 0
        
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": avg_chunk_size,
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap
        }


def main():
    """Example usage of the PDFProcessor class."""
    # Set your API key as environment variable (safer than hardcoding)
    
    
    # Initialize processor with much smaller chunk size to avoid token limits
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)  # Much smaller chunks
  
    
    # Configuration
    collection_name = "synthesis_papers"
    persist_directory = "docs/chroma_db"
    pdf_path = "docs/synthesis.pdf"
    
    # Process the synthesis PDF and store in ChromaDB
    try:
        # Check if ChromaDB collection already exists
        import chromadb
        client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            # Try to get the existing collection
            existing_collection = client.get_collection(name=collection_name)
            print(f"Found existing ChromaDB collection: {collection_name}")
            print(f"Collection has {existing_collection.count()} documents")
            
            # Load existing vectorstore
            vectorstore = processor.load_from_chromadb(
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            
        except Exception as e:
            # Collection doesn't exist, create it
            print(f"Collection not found, processing PDF and creating new collection...")
            vectorstore = processor.process_and_store_pdf(
                pdf_path, 
                collection_name=collection_name,
                persist_directory=persist_directory
            )

        # Debug: Check what documents are being retrieved for the question
        print("\n=== DEBUGGING: Checking retrieved documents ===")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Increased k to 8
        docs = retriever.get_relevant_documents("tell me about Origin of Dielectric Response")
        
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content preview: {doc.page_content[:400]}...")  # Show more content
            print("-" * 50)

        # Custom prompt for single, concise answer
        custom_prompt = PromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context. 
        The context contains information from a scientific document about polymer synthesis and dielectric properties.
        
        Instructions:
        1. Read the context carefully
        2. If the context contains information relevant to the question, provide a clear answer
        3. If the context doesn't contain relevant information, say "I don't know"
        4. Keep your answer to 5-6 sentences maximum
        5. Be specific and use information from the context

        Context:
        {context}

        Question: {question}
        
        Answer:""")

        # Use 'stuff' chain type for single answer
        qa_chain = RetrievalQA.from_chain_type(
            llm=processor.llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",  # Changed to stuff for single answer
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )

        # Test with a simpler question first
        print(f"\n=== TESTING SIMPLER QUESTION ===")
        simple_result = qa_chain.invoke({"query": "what is this document about?"})
        print(f"Simple question result: {simple_result['result']}")

        # Debug: Check what context is being passed
        print(f"\n=== DEBUGGING: Checking context passed to LLM ===")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.get_relevant_documents("tell me about Origin of Dielectric Response")
        
        # Combine context for debugging
        combined_context = "\n\n".join([doc.page_content for doc in context_docs])
        print(f"Context length: {len(combined_context)} characters")
        print(f"Context preview: {combined_context[:500]}...")

        print(f"\n=== QA CHAIN RESULT ===")
        simple_result = qa_chain.invoke({"query": "tell me about Origin of Dielectric Response"})
        print(f"Dielectric question result: {simple_result['result']}")
        
        # Also check the source documents used
        if "source_documents" in simple_result:
            print(f"\n=== SOURCE DOCUMENTS USED ===")
            for i, doc in enumerate(simple_result["source_documents"][:2]):  # Show first 2
                print(f"Source {i+1} - Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 