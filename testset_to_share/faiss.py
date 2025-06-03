from langchain.schema import Document
from docx import Document as DocxDocument
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

text_splitter = CharacterTextSplitter(
    separator='\n\n',  
    chunk_size=500,  
    chunk_overlap=1,  
    length_function=len 
)
all_chunks = []
csv_files = glob.glob('/content/*.csv')
csv_text = ""
for csv_path in csv_files:
    with open(csv_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            csv_text += " / ".join(row) + "\n"

    csv_chunks = text_splitter.split_text(csv_text)
    for i, chunk in enumerate(csv_chunks):
        doc = Document(page_content=chunk, metadata={"source_csv": csv_path, "chunk_index": i})
        all_chunks.append(doc)
    print(f"✅ CSVチャンク数: {len(csv_chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_chunks, embeddings)
vectorstore.save_local("faiss_store")
