import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "Vector In Memory"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama


if __name__ == "__main__":
    print("Hello Vector In Memory Project")
    pdf_path = "vector-in-memory/ReactPaper.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    print(f"Number of pages in the document: {len(document)}")
    print("Content of the first page:")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30,separator='\n')
    docs = text_splitter.split_documents(document)
    print(f"Number of chunks created: {len(docs)}")

    embeddings = OllamaEmbeddings(model='llama3.1:latest')
    #vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    #vectorstore.save_local("faiss_vectorstore")
    query = "Give me the gist of ReAct in 3 sentences"
    vectorstore = FAISS.load_local("faiss_vectorstore", embeddings,allow_dangerous_deserialization=True)
    llm = ChatOllama(model="llama3.1:latest", temperature=0, num_ctx=8000)
    combined_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=hub.pull("langchain-ai/retrieval-qa-chat"),
    )
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(),
        combined_docs_chain,
    )
    result = retrieval_chain.invoke(input={"input": query})
    print("Retrieval Result:", result["answer"])