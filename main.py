from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphSparqlQAChain
from langchain.graphs import RdfGraph
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

graph = RdfGraph(
    source_file="https://rdftest420.000webhostapp.com/card.rdf",
    standard="rdf",
    local_copy="test.ttl",
)

graph.load_schema()
print(graph.get_schema)

chain = GraphSparqlQAChain.from_llm(
    ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), graph=graph, verbose=True
)

print(chain.run("What is Tim Berners-Lee's work homepage?"))