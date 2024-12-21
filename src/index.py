from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

DOCUMENTS = "data"

def main() :
	print("Loading documents");
	loader = PyPDFDirectoryLoader(DOCUMENTS)

	docs = loader.load()
	docs = split(docs);
	print("Loaded");

	store(docs)
	print("Stored");


def store(docs:list[Document]) :
	db = Chroma(persist_directory="vectors", embedding_function=embeddingFunction())
	db.add_documents(docs)
	db.persist()


def split(docs:list[Document]) :
	text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=800,  # Desired chunk size
	chunk_overlap=80,  # Overlapping characters between chunks
	length_function=len,
	is_separator_regex=False)
	return(text_splitter.split_documents(docs))

def embeddingFunction() :
	embeddings = BedrockEmbeddings(
		credentials_profile_name="default", region_name="eu-central-1"
	)
	return(embeddings);


main()