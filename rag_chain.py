from langchain.chains import RetrievalQA

def create_rag_chain(vectorstore, llm):
    # Create the RetrievalQA chain with your LlamaCpp llm
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

