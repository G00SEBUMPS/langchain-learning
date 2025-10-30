from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import Pinecone
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import os
def upload_to_vector_store():
    loader = TextLoader("/home/vaibhav/Study/langchain/langchain-learning/vector-db/mediumarticle.md")
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("Splitting document into chunks...")
    print("Total number of documents before splitting:", len(documents))

    docs = text_splitter.split_documents(documents)
    print("Total number of documents after splitting:", len(docs))
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="llama3.1:latest")

    # Create Pinecone vector store
    vector_store = Pinecone.from_documents(
        docs,
        embeddings,
        index_name="test",
        
    )
    print("Vector store created successfully.")

def retrieve_from_vector_store():
    llm = ChatOllama(model="llama3.1:latest")
    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print("Result:", result.content)
    vector_store = Pinecone.from_existing_index(
        embedding=OllamaEmbeddings(model="llama3.1:latest"),
        index_name="test"
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt,
    )
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(),
        combine_docs_chain,
    )
    result = retrieval_chain.invoke(input={"input": query})
    print("Retrieval Result:", result["answer"])

if __name__ == "__main__":
    load_dotenv()
    print("Hello Vector DB Project")
    os.environ["LANGSMITH_PROJECT"] = "Vector DB Project"
    # Upload documents to vector store
    #upload_to_vector_store()
    # Retrieve from vector store
    retrieve_from_vector_store()
