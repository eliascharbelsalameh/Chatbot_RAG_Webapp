from langchain.prompts import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.chat_models import ChatOpenAI # type: ignore

def create_rag_chain(vectorstore, used_model):
    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.01, model_name=used_model),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain
