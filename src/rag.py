import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_aws import BedrockEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAG :
	_vectors:Chroma = None
	_history:list[str] = []

	__PROMPT__ = """
	Answer the question based only on the following context: {context}
	---
	Answer the question based on the above context: {question}
	"""

	limit:int = 8
	history:int = 16


	def __init__(self,location:str) :
		self._vectors = Chroma(persist_directory=location, embedding_function=RAG.__embeddingFunction())


	def query(self,text:str) :
		resp = ""
		sources = []

		for hist in self._history :
			resp += hist + "\n"

		try:
			docs = self._vectors.similarity_search_with_score(text,k=self.limit)
		except Exception as e:
			return("Error: " + str(e),sources)

		for tupl in docs :
			doc, score = tupl

			doc = doc.to_json()
			doc = doc.get("kwargs")
			mdata = doc.get("metadata")

			resp = "\n" + doc.get("page_content")

			source = RAG._to_json(mdata,score)
			sources.append(source)

		prmt = RAG.__PROMPT__.format(context=resp,question=text)

		model = OllamaLLM(model="mistral")

		try:
			response = model.invoke(prmt)
		except Exception as e:
			response = "Error: " + str(e)
			return(response,sources)

		self._history.append(text+"\n"+response)

		if len(self._history) > self.history :
			self._history.pop(0)

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


	def _to_json(mdata:json, score:int) :
		source = {}
		source["score"]  = score
		source["page"]   = mdata.get("page"),
		source["source"] = mdata.get("source")
		return(json.dumps(source))
