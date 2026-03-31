from dotenv import load_dotenv
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# load data
loader = DirectoryLoader("./docs", glob="*.txt")
documents = loader.load()
print("Documents loaded:", len(documents))

# split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
docs = splitter.split_documents(documents)
print("Chunks created:", len(docs))

# create embeddings
embeddings = OpenAIEmbeddings(openai_api_type=api_key)

#store in FAISS
db = FAISS.from_documents(docs, embeddings)

# save locally
db.save_local("faiss_index")
print("Embeddings stored in FAISS")

# LLM
llm = ChatOpenAI(api_key=api_key, temperature=0)

# RAG Function
def ask_rag(query):
    # step 1: retrieve relevant chunks
    db_docs = db.similarity_search(query, k=3)

    # step 2: combine context
    context = "\n".join([doc.page_content for doc in db_docs])
    print("context: ", context)

    # step 3: create a prompt
    prompt = f"""
You are a helpful assistant.

Rules:
- Answer only from the context below.
- If answer is not present, say "I don't know".

Context: {context}

Question: {query}
"""
    print("Sources: ")
    for i, doc in enumerate(db_docs):
        print(f"\nSource {i+1}:")
        print(doc.page_content)

    # step 4: get response
    response = llm.invoke(prompt)

    return response.content

# CLI loop
while True:
    query = input("Ask any question, (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = ask_rag(query)
    print("\nAnswer: ", answer)
