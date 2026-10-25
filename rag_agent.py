from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

# import os
# os.environ["LANGSMITH_PROJECT"]="rag_app"

# rag steps
# 1. parse the documents
# 2. give path and load the documents 
# 3. chunks the document
# 4. convert the chunks into embeddings
# 5. store the embeddings into vector database
# 6. retrive the relevant docs based on user query and give to llm
# 7. llm give the answer to user query

# 1. parse the documents>> not required as document in pdf
# 2. give path and load the documents
path = "transformer_paper.pdf"
loader = PyPDFLoader(path)
docs = loader.load()

# 3. chunks the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100,
    add_start_index = True
)
split_docs = text_splitter.split_documents(docs)

# 4. convert the chunks into embeddings
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

# 5. store the embeddings into vector database
vector_store = FAISS.from_documents(
    documents = split_docs,
    embedding = embeddings
)

model = init_chat_model(
    model = "gpt-4o-mini",
    max_tokens = 1000,
    temperature = 0,
    timeout = 120,
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

@tool
def retrieve_tool(query: str) -> str:
    """Search the research paper for information relevant to the user's query."""
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return "No relevant documents were found."

    return "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

prompt = (
    "You are a helpful research-paper assistant. "
    "Answer questions using information retrieved from the PDF. "
    "If the retrieved documents do not contain the answer, "
    "say that the information was not found in the document. "
    "Do not invent facts."
)

agent = create_agent(
    model = model,
    system_prompt = prompt,
    tools = [retrieve_tool],
)

result = agent.invoke(
    {"messages":[{"role":"user","content":"what is transformer"}]}
)

print(result["messages"][-1].content)



