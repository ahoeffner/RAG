import json
import logging
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from transformers import GPT2TokenizerFast
from langchain_aws import BedrockEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Rag :
	_history:list[str] = []
	_vectordb:Chroma = None

	__PROMPT__ = """
	Answer the question based only on the following context:

	{context}

	Answer the question based on the above context:

	{question}

	"""

	limit:int = 8
	history:int = 16


	def __init__(self,location:str) :
		self._vectordb = Chroma(persist_directory=location, embedding_function=Rag.__embeddingFunction())


	def query(self,text:str) :
		resp = ""
		hist = ""
		sources = []
		logger = Rag.logger()

		for prev in self._history :
			hist += prev + "\n"

		try:
			docs = self._vectordb.similarity_search_with_score(text,k=self.limit)
		except Exception as e:
			return("Error: " + str(e),sources)

		for tupl in docs :
			doc, score = tupl

			doc = doc.to_json()
			doc = doc.get("kwargs")
			mdata = doc.get("metadata")

			resp = "\n" + doc.get("page_content")

			source = Rag._to_json(mdata,score)
			sources.append(source)

			logger.debug(source)

		model = OllamaLLM(model="mistral")
		prmt = Rag.__PROMPT__.format(context=hist+"\n"+resp,question=text)

		try:
			response = model.invoke(prmt)
			logger.debug(response)
		except Exception as e:
			response = "Error: " + str(e)
			return(response,sources)

		self._history.append(text+"\n"+resp)

		if len(self._history) > self.history :
			self._history.pop(0)

		return(response,sources)


	def create(location:str, docs:list[Document]) :
		for doc in docs :
			doc = doc.to_json()
			doc = doc.get("kwargs")
			mdata = doc.get("metadata")
			score = doc.get("score")
			source = Rag._to_json(mdata,score)
			print(source)

		Chroma.from_documents(docs,Rag.__embeddingFunction(),persist_directory=location)


	def splitText(docs:list[Document]) :
		tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
		text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
			tokenizer,
			chunk_size=8000,
			chunk_overlap=512)
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


	def logger() :
		logging.basicConfig(
			filename="rag.log",
			filemode='a',
			format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
			datefmt='%H:%M:%S',
			level=logging.DEBUG)
		return(logging.getLogger("Rag"))