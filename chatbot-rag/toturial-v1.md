# TUTORIAL.md

## Building a Full-Stack Retrieval-Augmented Generation (RAG) Chatbot

This tutorial will guide you through the process of building a full-stack Retrieval-Augmented Generation (RAG) chatbot using FastAPI, OpenAI's language model, and Streamlit. By the end of this tutorial, you will have a working chatbot that can answer questions based on uploaded PDF documents.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- pip (Python package installer) or conda or uv

### Project Structure

Your project should have the following structure:

```plaintext
chatbot-rag
├── data/ #directory that hold loical victorial database
├── api.py
├── app.py
├── chatbot.py
├── requirements.txt
├── README.md
├── TUTORIAL.md
└── .env
```

### Step 1: Setting Up the Environment

1. **Create a virtual environment** (optional but recommended):

   1. using python virtualenv

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   2. using conda

   ```sh
   conda init
   conda create -n chatbot-rag python=3.11
   conda activate chatbot-rag
   ```

   3. using uv

   ```sh
   uv init
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

2. **Install the required dependencies**:

   Navigate to your project directory and run:
   for pip

   ```sh
   pip install -r requirements.txt
   ```

   for conda

   ```sh
   conda install -r requirements.txt
   ```

   for uv

   ```sh
   uv add -r requirements.txt
   uv sync
   ```

### Step 2: Configuring Environment Variables

Create a `.env` file in your project root and add your OpenAI API key:

```.env
OPENAI_API_KEY=your_openai_api_key
```

### Step 3: Building the Chatbot Logic

In `chatbot.py`, implement the core logic for handling documents and queries:
first we need to inmport the libraries that we are using

```python
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import PyPDFParser
import logging
```

to ensure that each error or info that our app make is logged and help us monitor our application we need to use the looging library

```python
# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

we are gouing to use langchain to create the embedding and llm object taht enables us to interact with openAI api but first we need to load the API_key from the .env file using dotenv librariy 'the APi keys are like secret that you never publish or share that why we keep them hidden in the .env file'

```python
# load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set")
    raise ValueError("OPENAI_API_KEY is not set")
# create OpenAI instance
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
```

every rag chatbot need a databse to save the document that uses as refrence but we can use any database because the rag system convert the documents into victors so it can srtores it and can preform similarity search to fertch the relevant document to the query that why we need a victor database like chorma which uses sqlite to store files victores localy qwhich is good for smalll project
the rag system use aretrivere that perform the search algorithm to fine the k relevant doment in our case we dicided to

```python
# setup Chroma database to store the documents
chroma = Chroma(
    collection_name="documents", # the collection name where we store group of document
    collection_metadata={"name": "documents", "description": "store documents"}, # extra information to define the collection
    persist_directory="./data", # path where to store the data
    embedding_function=embeddings, # embedding function that convert text into vector
)

# create a retriver to search the document
retriver = chroma.as_retriever(search_kwargs={"k": 2})
```

now we need to define the prompt that guide our llm to generate response from the context of relevant documents that we retrived. after that we create the chain that link the LLM with the retriver so after the retriver fertch relevent document and inject then into the prompt then after that it pass the prompt to the llm where he generate the answer to the question bassed on the provided documents

```python


# create a prompt template
TEMPLATE = """
Here is the context:

<context>
{context}
</context>

And here is the question that must be answered using that context:

<question>
{input}
</question>

Please read through the provided context carefully. Then, analyze the question and attempt to find a
direct answer to the question within the context.

If you are able to find a direct answer, provide it and elaborate on relevant points from the
context using bullet points "-".

If you cannot find a direct answer based on the provided context, outline the most relevant points
that give hints to the answer of the question. For example:

If no answer or relevant points can be found, or the question is not related to the context, simply
state the following sentence without any additional text:

i couldnt find an answer did not find an answer to your question.

Output your response in plain text without using the tags <answer> and </answer> and ensure you are not
quoting context text in your response since it must not be part of the answer.
"""

PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

# create the document parsing chain to inject the document into the chatbot
llm_chain = create_stuff_documents_chain(llm, PROMPT)

# create the the retrival chain
retrival_chain = create_retrieval_chain(retriver, llm_chain)
```

after that we use our chain to define our function that the api will intreact with we define 4 fuunction

- sotere document: a function that enable the api to store the documents into the database

```python

# create function to store the document into the database

def store_document(documents: list[Document]) -> str:
    """store the document into the database
    Args:
        documents (list[dict]): list of documents to store
    """
    chroma.add_documents(documents=documents)
    return "document stored successfully"
```

- parse pdf: a function that enable us to parse pdf files into docments that later can be stored int the database

```python
# ceate a pdf parser
parser = PyPDFParser()


def parse_pdf(file_content: bytes) -> list[Document]:
    """parse the pdf file
    Args:
        file_content (bytes): content of the pdf file
    """
    blob = Blob(data=file_content)
    document = [doc for doc in parser.lazy_parse(blob)]
    return document
```

- retrive document: a function that let the user search the database for relevant documents to given query

```python

#  create a function to retrive the document from the database
def retrieve_document(query: str) -> list[Document]:
    """retrieve the document from the database
    Args:
        query (str): query to search the document
    """
    documents = retriver.invoke(input=query)
    return documents
```

- ask question: a function that let the user ask the retrival chain a question about give query

```python
def ask_question(query: str) -> str:
    """chat with the chatbot
    Args:
        query (str): query to ask the chatbot
    """
    response = retrival_chain.invoke({"input": query})
    answer = response["answer"]
    return answer

```

### Step 4: Implementing the FastAPI Server

Open `api.py` and implement the FastAPI server:
fist we need to import the neccesarry libraries

```python
from fastapi import FastAPI, UploadFile
from chatbot import retrieve_document, store_document, parse_pdf, ask_question
from pydantic import BaseModel
from typing import List
import logging
```

as we showed earlier we will also implment the logger in the api for monitoring

```python
# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

we need teh fastapi to help us build the RestAPi appliaction that enables us to use HTTP protocl to send and recive data and act as gateway between our backend logic and the user interface

```python
# create FastAPI instance
app = FastAPI(
    title="Chatbot RAG",
    description="A simple chatbot using OpenAI. to enable asking questions and getting answers based on the uploaded document.",
    version="0.1",
)

# define the API endpoints
# define the root endpoint to check the status of the service
@app.get("/")
def read_root():
    return {
        "service": "RAG Chatbot using OPENAI",
        "description": "Welcome to Chatbot RAG API",
        "status": "running",
    }

```

to ensure that our data follow defined structure and types we need to use the pydantic and typing libraries to define the requests and the response class that enable us to make strict rules on user input and api output

```python
#  define the response and request models
class DocumentResponse(BaseModel):
    """Response model for the document API."""

    documents: List
    total: int
    query: str
    error: str = None


class DocumentUploadResponse(BaseModel):
    """Response model for the document upload API."""

    documents: List
    total: int
    status: str
    error: str = None


class AskResponse(BaseModel):
    """Response model for the ask API."""

    query: str
    answer: str
    error: str = None
```

well our api need to create 3 main endpoint to enable our application to use all the service:

- the first one is the is get endpoint that let the caller retrive the dfocuements trelative to the query

```python
#  define the document search endpoint to search documents based on the query
@app.get("/documents/{query}")
def search_documents(query: str) -> DocumentResponse:
    """Search documents based on the query."""
    try:
        documents = retrieve_document(query)
        return {"documents": documents, "total": len(documents), query: query}
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        return {"error": str(e), "documents": [], "total": 0, query: query}
```

- the second one is the upload document let the user uplad list of diffrent pdf files that we later will treat and save into our database

```python

# define the document upload endpoint to upload documents
@app.post("/documents")
async def upload_documents(files: List[UploadFile]) -> DocumentUploadResponse:
    """Store documents."""
    try:
        documents = []
        for file in files:
            if file.content_type != "application/pdf":
                logger.error(f"Unsupported file type: {file.content_type}")
                raise ValueError("Only PDF files are supported")
            content = await file.read()
            parsed_docs = parse_pdf(content)
            documents.extend(parsed_docs)
        status = store_document(documents)
        return {"documents": documents, "total": len(documents), "status": status}
    except Exception as e:
        logger.error(f"Error uploading documents: {e}", exc_info=True)
        return {"error": str(e), "status": "failed", "documents": [], "total": 0}


```

- the third is the the ask question that enable the user to ask a query and the ai chatbot return answer relative to the document in the database

```python
# define the ask endpoint to ask questions to the chatbot
@app.get("/ask")
def ask(query: str) -> AskResponse:
    """Ask questions to the chatbot."""
    try:
        answer = ask_question(query)
        return {"query": query, "answer": answer}
    except Exception as e:
        logger.error(f"Error asking question: {e}", exc_info=True)
        return {"error": str(e), "query": query, "answer": ""}

```

### Step 5: Creating the Streamlit Application

Open `app.py` and create the user interface: the user interface consist of way to upload pdf document so the use can later ask the ai assistanet about infornmation related to it that we have stored ibn the vectore database so we can use similarity search to retrive the document thta are relevant to the question

```python

import streamlit as st
import requests


# define the logic for the application
def ask(query: str) -> str:
    """ask the chatbot a question
    Args:
        query (str): question to ask the chatbot
    return:
        str: answer from the chatbot
    """
    with st.spinner("Asking the chatbot..."):
        response = requests.get(f"{API_URL}/ask?query={query}")

    if response.status_code == 200:
        data = response.json()
        return data["answer"]
    else:
        return "I couldn't find an answer to your question."


# define the base url for the API
API_URL = "http://localhost:8000"  # change this to the deployed API URL
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot RAG")

uploaded_files = st.file_uploader(
    "Upload you pdf docs", type="pdf", accept_multiple_files=True
)
if uploaded_files:
    files = [
        ("files", (file.name, file.getvalue(), "application/pdf"))
        for file in uploaded_files
    ]
    try:
        with st.spinner("Uploading files..."):
            response = requests.post(f"{API_URL}/documents/", files=files)
        if response.status_code == 200:
            st.success("Files uploaded successfully")
            uploaded_files = None
        else:
            st.error("Failed to upload files")
    except Exception as e:
        st.error(f"Error uploading files: {e}")


with st.chat_message(name="ai", avatar="ai"):
    st.write("Hello! I'm the Chatbot RAG. How can I help you today?")

query = st.chat_input(placeholder="Type your question here...")

if query:
    with st.chat_message("user"):
        st.write(query)
    answer = ask(query)
    with st.chat_message("ai"):
        st.write(answer)

```

### Step 6: Running the Application

1. **Start the FastAPI server**:

   ```sh
   uvicorn api:app --reload
   ```

2. **Run the Streamlit application**:

   ```sh
   streamlit run app.py
   ```

### Conclusion

Congratulations! You have successfully built a full-stack Retrieval-Augmented Generation (RAG) chatbot using FastAPI, OpenAI, and Streamlit. You can now upload PDF documents and interact with the chatbot to get answers based on the content of those documents.

### Author

- **LinkedIn**: [Eng. Oussama MAHDJOUR](https://www.linkedin.com/in/oussamamahdjour/)
- **Email**: [dev.mahdjour.oussama@gmail.com](mailto:dev.mahdjour.oussama@gmail.com)
- **GitHub**: [Oussama Mahdjour](https://github.com/mahdjourOussama)

### Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
