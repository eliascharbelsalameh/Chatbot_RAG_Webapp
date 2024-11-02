import os
import pdfplumber # type: ignore
import docx # type: ignore
import pandas as pd
from langchain.docstore.document import Document # type: ignore

def read_files_from_directory(directory_path):
    documents = []
    if not os.path.exists(directory_path):
        return documents

    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.startswith('.'):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    with pdfplumber.open(file_path) as pdf:
                        for page_number, page in enumerate(pdf.pages, start=1):
                            text = page.extract_text()
                            if text:
                                documents.append(Document(page_content=text, metadata={"source": filename, "page": page_number}))
                            tables = page.extract_tables()
                            for table in tables:
                                table_text = pd.DataFrame(table).to_string()
                                documents.append(Document(page_content=table_text, metadata={"source": filename, "page": page_number, "type": "table"}))
                                # log_debug(f"Extracted table from PDF: {filename}, Page: {page_number}")
                elif filename.lower().endswith('.docx'):
                    doc = docx.Document(file_path)
                    full_text = [para.text for para in doc.paragraphs]
                    documents.append(Document(page_content="\n".join(full_text), metadata={"source": filename}))
                elif filename.lower().endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                    documents.append(Document(page_content=df.to_string(), metadata={"source": filename}))
            except Exception as e:
                continue

    return documents
