from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

DOCUMENTS = "data"


def main() :
	'''
	print("Loading")
	loader = PyPDFDirectoryLoader(DOCUMENTS)
	docs = loader.load()

	docs = RAG.splitText(docs)
	RAG.create("database",docs)

	print("Stored");
	'''

	rag = RAG("database")
	docs:list[Document] = rag.query("can i specify return type in python")

	for doc in docs :
		print(doc.to_json().get("kwargs").get("page_content"))
		print()
		print()
		print()
		print()



class RAG :
	vectors:Chroma = None


	def __init__(self,location:str) :
		self.vectors =	Chroma(persist_directory=location, embedding_function=RAG.__embeddingFunction())


	def query(self,text:str) :
		return(self.vectors.similarity_search(text))


	def create(location:str, docs:list[Document]) :
		Chroma.from_documents(docs,RAG.__embeddingFunction(),persist_directory=location)


	def splitText(docs:list[Document]) :
		text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=800,  # Desired chunk size
		chunk_overlap=80,  # Overlapping characters between chunks
		length_function=len,
		is_separator_regex=False)
		return(text_splitter.split_documents(docs))

	def __embeddingFunction() :
		embeddings = BedrockEmbeddings(
			credentials_profile_name="default", region_name="eu-central-1"
		)
		return(embeddings)


main()