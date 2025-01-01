from Rag import Rag
from langchain_community.document_loaders import PyPDFDirectoryLoader

DOCUMENTS = "data"
LOCATION = "database"

def main() :
	print("Loading")
	loader = PyPDFDirectoryLoader(DOCUMENTS)
	docs = loader.load()

	Rag.create(LOCATION,Rag.splitText(docs))
	print("Stored in",LOCATION)

main()