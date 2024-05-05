from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from pprint import pprint

from documents import get_documents, split_docs
from vectorize import add_to_vectorstore
from llms import get_router
from llms import get_retreival_grader
from llms import get_generation
from llms import get_hallucination_grader
from llms import get_answer_grader
from llms import get_question_rewriter
from llms import get_websearch_tool

class RAGSystem:
    def __init__(self, llm_name):
        # Initialize your RAG system (e.g., load models, index documents, etc.)
        # This should happen only once during server startup

        
        self.llm_name = llm_name
        self.llm = ChatOllama(model=self.llm_name, format="json", temperature=0, base_url="http://host.docker.internal:11434")

        #(model="mixtral", verbose=False, temperature=0, base_url="http://ollama-container:11434")

        self.pdf_array = get_documents()
        self.list_of_docs = split_docs(self.pdf_array)
        self.retriever = add_to_vectorstore(self.list_of_docs)

        #LLMs
        self.question_router = get_router(self.llm)
        self.retrieval_grader = get_retreival_grader(self.llm)
        self.generation = get_generation(self.llm)
        self.hallucination_grader = get_hallucination_grader(self.llm)
        self.answer_grader = get_answer_grader(self.llm)
        self.question_rewriter = get_question_rewriter(self.llm)

        self.web_search_tool = get_websearch_tool(self)



    def get_pdf_array(self):
        return self.pdf_array
    
    def get_list_of_docs(self):
        return self.list_of_docs


    def get_retreiver(self):
        return self.retriever
    

    def get_router(self):
        return self.question_router
    
    def get_retreival_grader(self):
        return self.retrieval_grader

    def get_generation(self):
        return self.generation
         
         
    def get_hallucination_grader(self):
        return self.hallucination_grader

    def get_answer_grader(self):
        return self.answer_grader

    def get_question_rewriter(self):
        return self.question_rewriter

    def get_websearch_tool(self):
        return self.web_search_tool


    async def generate_answer(self, app, question: str) -> str:
        # Process the question and generate an answer
        # You can reuse the same instance for multiple requests
        # (e.g., use cached data, precomputed indexes, etc.)
        inputs = {"question": question}
        print(inputs)
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                #print.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")
        # Final generation
        #pprint(value["generation"])
        return value["generation"]


    

    ### Nodes

    def retrieve(self,state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        print(question)
        # Retrieval
        #documents = self.retriever.get_relevant_documents(question)
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # RAG generation
        rag_chain = self.generation.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": rag_chain}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    ### Edges ###

    def route_question(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)
        source = self.question_router.invoke({"question": question})  
        print(source)
        print(source['datasource'])
        if source['datasource'] == 'web_search':
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source['datasource'] == 'vectorstore':
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question,"generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"