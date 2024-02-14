from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphSparqlQAChain
from langchain.graphs import RdfGraph
from dotenv import load_dotenv
import os
from rdflib import Graph

# g = Graph()
# g.parse("http://raw.githubusercontent.com/c0pper/sparql_langchain/main/card.rdf", format='xml')
# print(len(g))
# import pprint
# for stmt in g:
#     pprint.pprint(stmt)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

graph = RdfGraph(
    source_file="uni.rdf",
    standard="rdfs",
    serialization='xml',
    local_copy="test.rdf",
)

graph.load_schema()
print(graph.get_schema)

chain = GraphSparqlQAChain.from_llm(
    ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), graph=graph, verbose=True
)
q = "Which researchers employed at Università di Napoli collaborate with Università di Messina?"
print(f"Q: {q}")
print(chain.run(q))