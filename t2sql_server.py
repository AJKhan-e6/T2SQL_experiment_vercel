import json
import logging
import sys
import time
import uvicorn
from getsql import get_top_matches
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    JSONLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import MarkdownTextSplitter
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import sqlalchemy

OPENAI_ACCESS_TOKEN = os.getenv('OPENAI_API_KEY')
if not OPENAI_ACCESS_TOKEN:
    raise ValueError("OpenAI API key not found. Set it in the environment variables.")
model = "gpt-4o"

PORT = 8010



logging.basicConfig(
    level=logging.INFO, format="Server: [%(asctime)s] %(levelname)s %(message)s"
)
Logger = logging.getLogger()
app = FastAPI(
    title="Openai",
    description="Openai",
    version="1.0.0",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    debug=True,
)

class UserQuery(BaseModel):
    question: str


class LangChainService:
    _service = None
    history_chain = None

    @staticmethod
    def get_history_chain():
        return LangChainService.get_singleton_instance().history_chain

    @classmethod
    def get_singleton_instance(self):
        """
        Creates the Service object if it is not yet created,
        otherwise uses the already created object.
        """
        if self._service is None:
            print("creating singleton service")
            self._service = self()
            self.history_chain = self._service.load_history_chain()
            print("Created chat engine...")
        else:
            print("Retruned existing obj without timedelay")
        return self._service

    def load_history_chain(self):

        # Setting up the documentation path
        document_path_supp = "./supported-sql-functions"
        document_path_eq = "./equivalent-sql-functions"

        # Setting up the prompt directives
        directives_file = open("directives_v4.txt", "r")
        directives = directives_file.read()

        # Get an OpenAI API Key before continuing
        openai_api_key = OPENAI_ACCESS_TOKEN

        # Setting up the document indexing
        docs = DirectoryLoader(
            document_path_supp,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            use_multithreading=True,
            recursive=True,
        ).load()
        split_text = MarkdownTextSplitter(
            chunk_size=128, chunk_overlap=60
        ).split_documents(docs)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large", api_key=openai_api_key
        )
        db = Chroma.from_documents(split_text, embedding_model)
        retriever_sp = db.as_retriever(k=8)

        # Making a retriever tool
        tool_sp = create_retriever_tool(
            retriever_sp,
            "supported_functions",
            "Contains all functions supported by e6data, along with syntax and examples.",
        )

        # Setting up the document indexing
        docs_eq = DirectoryLoader(
            document_path_eq,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            use_multithreading=True,
            recursive=True,
        ).load()
        split_text_eq = MarkdownTextSplitter(chunk_size=4100).split_documents(docs_eq)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large", api_key=openai_api_key
        )
        db_eq = Chroma.from_documents(split_text_eq, embedding_model)
        retriever_eq = db_eq.as_retriever(k=2)
        # Making a retriever tool
        tool_eq = create_retriever_tool(
            retriever_eq,
            "equivalent_functions",
            "Contains functions which are the e6data equivalents of those in other engines.",
        )

        tools = [tool_sp, tool_eq]

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"Your job is to write SQL queries given a user's request and correct errors in SQL queries.\n\n{directives}\n\n. Use the retreiver_sp and retriever_eq tools to find the correct supported functions and use them in the query.\nAlways return the line numbers where changes have to be made in the original SQL query. Place special emphasis on what the user is asking in their question.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        history_for_chain = ChatMessageHistory()

        # Chat agent core
        chat = ChatOpenAI(
            temperature=0.2, api_key=openai_api_key, model=model, timeout=600
        )
        agent = create_openai_tools_agent(llm=chat, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Making a runnable memory chain
        chain_with_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: history_for_chain,
            input_messages_key="question",
            output_messages_key="output",
            history_messages_key="history",
        )
        return chain_with_history


def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)):
        return size + sum(map(sizeof, obj))
    return size


def format_sql_query(query):
    try:
        # Parse the SQL query
        parsed = sqlalchemy.text(query)

        # Format the query
        formatted_query = str(parsed.compile(compile_kwargs={"literal_binds": True}))

        # Split the formatted query into lines and add line numbers
        lines = formatted_query.split("\n")
        numbered_lines = []
        for i, line in enumerate(lines, start=1):
            numbered_lines.append(f"\n{i:>4}: {line}")

        return "\n".join(numbered_lines)
    except Exception as e:
        return "No valid SQL entered"


@app.get("/healthz")
async def healthz():
    return dict(status="Success")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = round(time.time() - start_time, 2)
    Logger.info(f"Server side execution time {process_time}s.")
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.post("/")
async def test(request: UserQuery):
    # Extract user question from the request body
    user_question = request.question

    # Log the user question
    Logger.info(f"User Question: {user_question}")

    # Get the top matches using the `get_top_matches` function
    try:
        top_matches = get_top_matches(user_question, k=3)
    except Exception as e:
        Logger.error(f"Error fetching top matches: {e}")
        top_matches = []

    # Format the top matches as examples for the prompt
    example_queries = []
    for match in top_matches:
        example_question = match.get('Natural Language Query', '')
        sql_query = match.get('SQL Query', '')
        filtered_schema = match.get('Filtered Schema', '')
        example_queries.append(f"Similar Question: {example_question}\nExample SQL: {sql_query}\n\nSchema: {filtered_schema}\n")

    # Prepare the prompt
    examples_text = "\n\n".join(example_queries)
    prompt = f"User Question: {user_question}\n\nExamples:\n{examples_text}" if example_queries else user_question

    Logger.info(f"Formatted Prompt: {prompt}")

    # Call LangChainService
    history_chain = LangChainService.get_history_chain()
    config = {"configurable": {"session_id": "any"}}
    response = history_chain.invoke({"question": prompt}, config)


    # Extract and return the 'output' section
    output = response.get("output", "")
    return {"output": output}


@app.on_event("startup")
async def startup():
    Logger.info(f"Server started... ")
    LangChainService.get_singleton_instance()
    Logger.info(f"Loaded index... ")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        # log_level=LOG_LEVEL,
        proxy_headers=True,
        workers=1,
    )
