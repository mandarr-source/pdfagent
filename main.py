from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

memory = MemorySaver()
model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")


@tool
def multiply(a: int, b: int) -> int:
    """multiply two number."""
    return a * b


pdf_path = "~/Desktop/agents/Account_stmt (1).pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma vector store
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def search_pdf(query):
    """search in pdf for the query provided"""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])


search = TavilySearchResults(max_result=2)
tools = [search, multiply, search_pdf]
agent_executor = create_react_agent(
    model,
    tools,
    prompt="Keep the output minimal",
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="add all the numbers in the pdf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()