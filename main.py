import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from jira import JIRA
from config.settings import *
import sqlite3


# Load CSV data from settings
def load_data():
    # data = pd.read_csv(CSV_FILE_PATH)
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql_query(SELECT_QRY, conn)

    # Ensure the Positive_Counter and Negative_Counter columns exist
    if 'Positive_Counter' not in data.columns:
        data['Positive_Counter'] = 0
    if 'Negative_Counter' not in data.columns:
        data['Negative_Counter'] = 0
    
    return data

# Create or load FAISS vector store for semantic search
@st.cache_data(show_spinner=False)
def create_faiss_vectorstore(df):
    documents = [
        Document(
            page_content=f"Issue Type: {row['Issue_Type']}\nRCA: {row['RCA']}\nSteps to Resolve: {row['Steps_to_Resolve']}",
            metadata={"Issue Type": row["Issue_Type"], "RCA": row["RCA"], "Steps to Resolve": row["Steps_to_Resolve"]}
        )
        for _, row in df.iterrows()
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Create vector store using FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# Function to search the FAISS vector store
def search_faiss_vectorstore(query, vectorstore, threshold=0.75):
    results = vectorstore.similarity_search_with_score(query)
    if (results and results[0][1] >= threshold) and (results and results[0][1] <= 1):  # Compare with the threshold
        top_result = results[0][0]  # Get the top result document
        return top_result.metadata, results[0][1]  # Return metadata and score
    else:
        return False, False

# Function to create JIRA ticket if no issue found
def create_jira_ticket(summary, description):
    jira_options = {'server': JIRA_URL}
    jira = JIRA(options=jira_options, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))

    issue_dict = {
        'project': {'key': JIRA_PROJECT_KEY},
        'summary': summary,
        'description': description,
        'issuetype': {'name': JIRA_ISSUE_TYPE},
    }

    new_issue = jira.create_issue(fields=issue_dict)
    return new_issue
# Update the "Positive_Counter" or "Negative_Counter" based on user feedback
def update_feedback_counter(df, issue_type, rca, feedback_type):
    idx = df[(df['Issue_Type'] == issue_type) & (df['RCA'] == rca)].index
    if not idx.empty:
        idx = idx[0] 
        if feedback_type == 'Positive':
            df.at[idx, 'Positive_Counter'] += 1
        elif feedback_type == 'Negative':
            df.at[idx, 'Negative_Counter'] += 1
        df.to_csv(CSV_FILE_PATH, index=False)

    # Connect to SQLite database
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # Update the SQLite table
    if feedback_type == 'Positive':
        cursor.execute('''
            UPDATE data
            SET Positive_Counter = Positive_Counter + 1
            WHERE Issue_Type = ? AND RCA = ?
        ''', (issue_type, rca))
    elif feedback_type == 'Negative':
        cursor.execute('''
            UPDATE data
            SET Negative_Counter = Negative_Counter + 1
            WHERE Issue_Type = ? AND RCA = ?
        ''', (issue_type, rca))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()


# Frontend: Continuous Chat Interface
st.title(TITLE)

# Load CSV data
data = load_data()

# Use session state to avoid reloading the CSV multiple times
if 'df' not in st.session_state:
    st.session_state.df = data

df = st.session_state.df

# Precompute the FAISS vector store
vectorstore = create_faiss_vectorstore(df)

# Initialize session state for storing messages (chat history)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None  # To track the last matched issue for feedback

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.chat_message('user').markdown(message['content'])
    elif message['role'] == 'assistant':
        st.chat_message('assistant').markdown(message['content'])

# User input for continuous chat
user_query = st.chat_input("Ask your IT question here...")

if user_query:
    # Store the user's message in session state
    st.chat_message('user').markdown(user_query)
    st.session_state.messages.append({'role': 'user', 'content': user_query})
    
    # Search the FAISS vector store
    result, score = search_faiss_vectorstore(user_query, vectorstore, threshold=0.75)

    if result:
        response = f"**Matched Issue Type:** {result['Issue Type']}\n\n**RCA:** {result['RCA']}\n\n**Steps to Resolve:** {result['Steps to Resolve']}"
        
        # Store the assistant's response in session state
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.session_state.last_result = result 
        # Display the assistant's response
        st.chat_message('assistant').markdown(response)
        feedback = st.radio("Was this helpful?", ('👍', '👎'),horizontal=True,index=None)

        # Update the counters based on feedback
        if feedback == '👍':
            update_feedback_counter(df, result['Issue_Type'], result['RCA'], 'Positive')
        elif feedback == '👎':
            update_feedback_counter(df, result['Issue_Type'], result['RCA'], 'Negative')

       
    else:
        # If no match is found, provide an alternate response
        st.session_state.messages.append({'role': 'assistant', 'content': "No matching issue found. Creating a JIRA ticket..."})
        st.chat_message('assistant').markdown("No matching issue found. Creating a JIRA ticket...")
        # Create a JIRA ticket
        summary = f"New issue reported: {user_query}"
        description = f"A user reported the following issue:\n\n{user_query}"
        # new_issue = create_jira_ticket(summary, description)

        # # Display the JIRA ticket details
        # ticket_message = f"JIRA Ticket Created: [{new_issue.key}]({JIRA_URL}/browse/{new_issue.key})"
        # st.session_state.messages.append({'role': 'assistant', 'content': ticket_message})
        # st.chat_message('assistant').markdown(ticket_message)