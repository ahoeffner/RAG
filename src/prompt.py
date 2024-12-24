from rag import RAG


LOCATION = "database"


def main() :
	rag = RAG(LOCATION)

	while(True) :
		try: text = input("Enter a query: ")
		except EOFError : break
		except KeyboardInterrupt : break

		if (text.__len__() == 0) :
			continue

		if(text.lower() == "exit") :
			break

		docs = rag.query(text)

		for doc in docs :
			print(doc.to_json().get("kwargs").get("page_content"))
			print()
			print()


main()