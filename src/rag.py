from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_aws import BedrockEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAG :
	vectors:Chroma = None

	PROMPT = """
	Answer the question based only on the following context: {context}
	---
	Answer the question based on the above context: {question}
	"""


	def __init__(self,location:str) :
		self.vectors =	Chroma(persist_directory=location, embedding_function=RAG.__embeddingFunction())


	def query(self,text:str) :
		resp = ""
		sources = []

		docs = self.vectors.similarity_search_with_score(text,k=5)

		for tupl in docs :
			doc, score = tupl

			doc = doc.to_json()
			doc = doc.get("kwargs")
			mdata = doc.get("metadata")

			resp = "\n" + doc.get("page_content")
			sources.append({mdata.get("source"),mdata.get("page"),score})

		prmt = RAG.PROMPT.format(context=resp,question=text)

		model = OllamaLLM(model="mistral")
		response = model.invoke(prmt)

		return(response,sources)


	def create(location:str, docs:list[Document]) :
		for doc in docs :
			print(doc.to_json())
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
