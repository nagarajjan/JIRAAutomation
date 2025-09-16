# JIRAAutomation
Here is a detailed, step-by-step process for building and deploying this application.

# Phase 1: Data ingestion and the RAG pipeline
Configure a Jira API key
Log in to your Atlassian account (https://id.atlassian.com/manage-profile/security/api-tokens).
Create a new API token with appropriate read/write scopes for Jira access. Store this token securely.
Select an ingestion method
Manual/scripted pull: Write a Python script using the Jira REST API or a library like jira-python to extract issue data (summary, description, comments, etc.) and save it as JSON files.
Automated connector: Use a tool like Unstructured.io or PyAirbyte, which offer pre-built Jira connectors to automate data extraction and processing.
Create vector embeddings
Clean and chunk data: Use a text splitter to break down large Jira issues and comments into smaller, manageable chunks.
Generate embeddings: Use an embedding model (like OpenAI's or a hosted alternative) to convert the text chunks into vector embeddings.
Populate the vector database
Choose a vector database (e.g., Pinecone or Qdrant).
Connect your application to the database and upload the vector embeddings. This creates the knowledge base that the RAG system will search. 
End-to-end RAG using Jira, PyAirbyte, Pinecone, and ...
This notebook demonstrates an end-to-end Retrieval-Augmented Generation (RAG) pipeline. We will extract data from Jira using PyAirbyte, store it in a Pinecone v...

Favicon

Airbyte

Step-by-Step Guide to Building a RAG-Powered JIRA ...

# Step 1: Setting Up the Flask Web Application. First, we'll set up a Flask web application to serve as the interface for querying our indexed JIRA tickets. Cod...

Favicon
Futurify
# Phase 2: Building the MCP server
Set up the server

Create a new Node.js or Python project.

Use a framework like Express.js (Node.js) or Flask (Python) to build a web server.

Install the MCP SDK (@modelcontextprotocol/sdk) and a Jira library (e.g., jira.js for Node.js).

Define Jira actions
In the server code, implement functions for each Jira action you want to expose to the LLM (e.g., createIssue, searchIssues, getIssueDetails).
These functions will use your Jira API key to authenticate and make requests to Jira's REST API.

Expose tools via MCP

Create an MCP tool manifest that describes the actions available in your server.

The MCP server will listen for requests and translate LLM instructions into Jira API calls. 

# Phase 3: Implementing the LLM and orchestration

Select an LLM agent framework
Use a framework like LangChain or LlamaIndex, which provide agents and tool orchestration capabilities.

This framework will manage the flow of logic: receiving a user query, deciding which tools to use, and generating a final response.

Connect to the RAG system

Configure the LLM agent to access your vector database. When a query is made, the agent will use the RAG system to retrieve relevant Jira tickets and use them as context.

Connect to the MCP server

Configure the agent to access your custom-built MCP server. This allows the LLM to call your defined Jira actions when needed. 
# Phase 4: Building the dashboard and a natural language interface

Create a user interface

Build a user interface using a web framework like Streamlit, which simplifies the creation of data-driven web apps.

Include a text input field for the user to type natural language commands.

Implement the dashboard logic

When a user submits a natural language request (e.g., "Show me the top 5 projects with the most open bugs"), the application will:

Send the query to the LLM agent.

The agent uses RAG to retrieve relevant project data.

The agent calls the MCP server to search Jira using JQL for the required data.

The application processes the retrieved data.

Streamlit renders an automatically generated dashboard or charts based on the data analysis.

Deploy the application

Package your application using Docker for consistent deployment.

Deploy the containers to a cloud service (e.g., AWS, GCP) or your own infrastructure. Ensure your LLM and vector database are also correctly configured and accessible. 

Example workflow

User query (via UI): "Show me the top 5 projects with the most open bugs."

Agent orchestration: The LLM agent receives the query and uses its understanding of the Jira domain to form a search plan.

RAG: The agent first searches the vector database to retrieve historical Jira data for context, ensuring the LLM understands how "bugs" and "top projects" are historically defined in your company's Jira.

MCP tool call: The agent executes a tool call to the MCP server, passing a Jira Query Language (JQL) command. The server translates this into an API call to Jira.

# Step 1: Set up the Python project
Create a project directory and install the necessary libraries. This project will contain the RAG system and the Streamlit frontend.
sh
mkdir jira_automation_app
cd jira_automation_app
pip install -r requirements.txt

# requirements.txt
streamlit
langchain
langchain-openai
pinecone-client
jira
unstructured[jira]
python-dotenv

# Step 2: Ingest Jira data for RAG
Create a script to connect to Jira, extract data, and populate your vector database. You will need to set up a free account with Pinecone to get your API key and environment.
# ingest_data.py
python

import os

from unstructured.ingest.connector.jira import JiraSourceConnector

from unstructured.ingest.connector.jira import SimpleJiraConfig

from unstructured.ingest.runner.jira import JiraRunner

from langchain.vectorstores import Pinecone

from pinecone import Pinecone as PineconeClient, ServerlessSpec

from langchain.embeddings import OpenAIEmbeddings

import logging

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

JIRA_URL = os.getenv("JIRA_URL")

JIRA_EMAIL = os.getenv("JIRA_EMAIL")

JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

# --- Ingestion ---
 def ingest_jira_data():
 
    """Ingests Jira data and stores it in the vector database."""
    
    logging.info("Starting Jira data ingestion...")
    
    jira_config = SimpleJiraConfig(
        url=JIRA_URL,
        username=JIRA_EMAIL,
        password=JIRA_API_TOKEN,
        projects=[PROJECT_KEY],
    )
    
    runner = JiraRunner(
        connector_config=jira_config,
        verbose=True,
    )
    
    jira_docs = runner.run()
    logging.info(f"Retrieved {len(jira_docs)} documents from Jira.")
    
    # Process documents and create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Initialize Pinecone
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index_name = "jira-rag-index"
    if index_name not in pc.list_indexes().names:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI's embedding size
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    
    pinecone_db = Pinecone.from_documents(jira_docs, embeddings, index_name=index_name)
    logging.info("Successfully ingested documents into Pinecone.")
    return pinecone_db

 if __name__ == "__main__":

    if not all([PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY, JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, PROJECT_KEY]):

        logging.error("Please set all required environment variables.")
    
    else:
        ingest_jira_data()

# Step 3: Create the MCP server (Express.js)
This Node.js server acts as the secure interface for the LLM to interact with Jira. It is a critical component for performing automated actions.

 mcp_jira_server/package.json

 json

{
 
  "name": "mcp_jira_server",
  
  "version": "1.0.0",
  
  "description": "An MCP server for Jira actions",
  
  "main": "index.js",
  
  "scripts": {
  
    "start": "node index.js"
  
  },
  
  "dependencies": {
  
    "@atlassian/jira": "^10.15.1",
    
    "@modelcontextprotocol/sdk": "^1.0.0",
    
    "express": "^4.19.2",
    
    "dotenv": "^16.4.5"
  
  }
}


# mcp_jira_server/index.js

javascript

import express from 'express';

import { Server } from '@modelcontextprotocol/sdk';

import { JiraClient } from '@atlassian/jira';

import dotenv from 'dotenv';

dotenv.config();

const app = express();

const port = 3000;

// Initialize Jira client

 const jiraClient = new JiraClient({
 
    host: process.env.JIRA_HOST,
 
    email: process.env.JIRA_EMAIL,
    
    token: process.env.JIRA_API_TOKEN,
 
 });

const server = new Server({
 
    name: "JiraMCP",
    
    version: "1.0.0",
    
description: "An MCP server for Jira automation",

});

// Define and expose a 'createIssue' tool

server.addTool({

    name: 'createIssue',
    
    description: 'Creates a new Jira issue.',
    
    handler: async (args) => {
    
        try {
        
            const issue = await jiraClient.issues.create(args);
            
            return `Successfully created Jira issue: ${issue.key}`;
        
        } catch (error) {
            return `Error creating Jira issue: ${error.message}`;
        
        }
    },

});

// Expose a 'searchIssues' tool

server.addTool({

    name: 'searchIssues',
 
    description: 'Searches for Jira issues using JQL.',
    
    handler: async (args) => {
    
        try {
        
            const issues = await jiraClient.issues.search(args.jql);
           
            return JSON.stringify(issues);
        
        } catch (error) {
        
            return `Error searching Jira issues: ${error.message}`;
        
        }
    
    },

});

app.use('/mcp', server.express());

app.listen(port, () => {

    console.log(`MCP server for Jira listening on port ${port}`);

});

# Step 4: Create the Streamlit application
This is the user interface where you will interact with the LLM agent. It combines the RAG context and the MCP tools to execute commands.

streamlit_app.py

python

import streamlit as st

import os

from langchain.vectorstores import Pinecone

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI

from langchain.agents import AgentExecutor, create_react_agent

from langchain.tools import Tool

from langchain_core.prompts import PromptTemplate

from pinecone import Pinecone as PineconeClient

# Load environment variables

from dotenv import load_dotenv

load_dotenv()

# --- Configuration and Setup ---

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

JIRA_MCP_URL = "http://localhost:3000/mcp"


# Connect to Pinecone for RAG

pc = PineconeClient(api_key=PINECONE_API_KEY)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone_db = Pinecone.from_existing_index("jira-rag-index", embeddings)

retriever = pinecone_db.as_retriever()

# Initialize LLM

llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Define tools for the LangChain agent

 def create_jira_issue(args):

    """MCP tool for creating a Jira issue."""
    
    import requests
    
    response = requests.post(f"{JIRA_MCP_URL}/createIssue", json={"args": args})
    
    return response.text

 def search_jira_issues(jql):
    
    """MCP tool for searching Jira issues with JQL."""
    
    import requests
    
    response = requests.post(f"{JIRA_MCP_URL}/searchIssues", json={"args": {"jql": jql}})
    
    return response.json()

tools = [

    Tool(
    
        name="create_jira_issue",
        
        func=create_jira_issue,
        
        description="Creates a new Jira issue with a given summary, description, and issue type."
    
    ),
    
    Tool(
    
        name="search_jira_issues",
        
        func=search_jira_issues,
        
        description="Searches Jira issues using JQL and returns the result as a JSON string."
    
    )
]

# Create the LangChain agent

prompt_template = PromptTemplate.from_template("""

You are a helpful AI assistant for managing Jira tickets. You have access to a RAG system

for relevant context and tools for interacting with Jira via an MCP server.

Use the RAG system to retrieve relevant information before attempting to answer a query.

If the user's query requires an action in Jira, use the appropriate tool.

Your goal is to provide accurate, helpful, and automated responses.

User Query: {input}

Context from RAG: {rag_context}

""")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Streamlit UI ---

st.title("Automated Jira Assistant")

if 'chat_history' not in st.session_state:
   
    st.session_state.chat_history = []

user_query = st.text_input("Enter your request:")

if st.button("Submit"):

    if user_query:
    
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Retrieve context from RAG
        
        rag_context = retriever.get_relevant_documents(user_query)
        
        # Format the query with RAG context
        
        final_query = prompt_template.format(input=user_query, rag_context=rag_context)

        # Run the agent
        
        try:
            response = agent_executor.invoke({"input": final_query})
        
            st.session_state.chat_history.append({"role": "assistant", "content": response.get('output', 'Error processing request.')})
        
        except Exception as e:
        
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {e}"})

for message in st.session_state.chat_history:
   
    with st.chat_message(message["role"]):
    
        st.write(message["content"])

# Step 5: Run the application

Start the MCP Server:

sh

cd mcp_jira_server

npm install

npm start

# Start the Streamlit App:

sh

cd ..

streamlit run streamlit_app.py

Jira search: Jira returns a list of issues matching the JQL query.
Dashboard generation: The application processes the Jira data and uses a plotting library (e.g., Plotly, Matplotlib) to generate a bar chart or table and renders it on the Streamlit dashboard. 
