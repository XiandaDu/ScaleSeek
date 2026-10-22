"""Methods implemented in this repository (GrepSeek and AgentIR wrap official harnesses
and live in grepseek.py / agentir.py)."""

from .direct import Direct
from .rag import RAG
from .search_o1 import SearchO1
from .search_r1 import SearchR1

METHODS = {"direct": Direct, "rag": RAG, "search_r1": SearchR1, "search_o1": SearchO1}
