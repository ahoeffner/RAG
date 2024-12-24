from src.rag import RAG
from langchain_community.document_loaders import PyPDFDirectoryLoader

DOCUMENTS = "data"
LOCATION = "database"

def main() :
	print("Loading")
	loader = PyPDFDirectoryLoader(DOCUMENTS)
	docs = loader.load()

	RAG.create(LOCATION,RAG.splitText(docs))
	print("Stored in",LOCATION)

main()