from typing import Union
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from documents import get_documents, split_docs
from vectorize import add_to_vectorstore
from llms import get_router

from rag_system_module import RAGSystem
from graph_state_module import GraphState

import os

app = FastAPI()

# Ollama model name
local_llm = "llama2"
run_local = "Yes"


os.environ['TAVILY_API_KEY'] = "[ADD YOUR API KEY HERE]"


# Instantiate your RAG system once (outside the route handlers)
rag_instance = RAGSystem(local_llm)

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", rag_instance.web_search) # web search
workflow.add_node("retrieve", rag_instance.retrieve) # retrieve
workflow.add_node("grade_documents", rag_instance.grade_documents) # grade documents
workflow.add_node("generate", rag_instance.generate) # generatae
workflow.add_node("transform_query", rag_instance.transform_query) # transform_query

# Build graph
workflow.set_conditional_entry_point(
    rag_instance.route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    rag_instance.decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    rag_instance.grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
graph = workflow.compile()



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/ask/{query}")
async def ask_question(query: str):
    answer = await rag_instance.generate_answer(graph, query)
    return {"answer": answer}