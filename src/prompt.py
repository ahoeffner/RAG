from Rag import Rag
from langchain.schema.document import Document


LOCATION = "database"


def main() :
	rag = Rag(LOCATION)

	while(True) :
		try: text = input("Enter a query: ")
		except EOFError : break
		except KeyboardInterrupt : break

		if (text.__len__() == 0) :
			continue

		if(text.lower() == "exit") :
			break

		response, sources = rag.query(text)

		print()
		print()
		print(response)
		print()
		print()


main()