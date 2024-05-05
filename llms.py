from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

def get_router(llm):
    ### Router

    # LLM
    #llm = ChatOllama(model=local_llm, format="json", temperature=temp)

    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no premable or explaination. \n
        Question to route: {question}""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()
    
    #if(question != ''):
        #question = "in context learning (ICL)"
        #docs = retriever.get_relevant_documents(question)
        #doc_txt = docs[1].page_content
    #    print(question_router.invoke({"question": question}))   

    return question_router

def get_retreival_grader(llm):

    # LLM
    #llm = ChatOllama(model=llm_name, format="json", temperature=temp)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()

    return retrieval_grader


def get_generation(llm):

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    #llm = ChatOllama(model=llm_name, temperature=temp)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain

def get_hallucination_grader(llm):

    # LLM
    #llm = ChatOllama(model=llm_name, format="json", temperature=temp)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

    return hallucination_grader


def get_answer_grader(llm):
    # LLM
    #llm = ChatOllama(model=llm_name, format="json", temperature=temp)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()

    return answer_grader

def get_question_rewriter(llm):

    # LLM
    #llm = ChatOllama(model=llm_name, temperature=temp)

    # Prompt 
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    return question_rewriter


def get_websearch_tool(self):
    return TavilySearchResults(k=3)