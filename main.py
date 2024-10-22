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
            page_content=f"Issue Type: {row['Summary']}\nRCA: {row['Custom_field_Root_Cause']}\nSteps to Resolve: {row['Comment']}",
            metadata={"Issue Type": row["Summary"], "RCA": row["Custom_field_Root_Cause"], "Steps to Resolve": row["Comment"]}
        )
        for _, row in df.iterrows()
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Create vector store using FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# Function to search the FAISS vector store
def search_faiss_vectorstore(query, vectorstore, threshold=0.60, top_k=3):
    results = vectorstore.similarity_search_with_score(query)
    filtered_results = [
        (doc, score) for doc, score in results if threshold <= score <= MAX_THRESHOLD  
    ]  # Filter results based on threshold
    filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
    # Return the top_k results
    return filtered_results[:top_k] if filtered_results else []

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
    # Update the CSV file
    idx = df[(df['Issue_Type'] == issue_type) & (df['RCA'] == rca)].index
    if not idx.empty:
        idx = idx[0]
        if feedback_type == 'Positive':
            df.at[idx, 'Positive_Counter'] += 1
        elif feedback_type == 'Negative':
            df.at[idx, 'Negative_Counter'] += 1
        df.to_csv(CSV_FILE_PATH, index=False)

    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
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

if 'expanded_results' not in st.session_state:
    st.session_state.expanded_results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'df' not in st.session_state:
    st.session_state.df = data

df = st.session_state.df

# Precompute the FAISS vector store
vectorstore = create_faiss_vectorstore(df)

# Initialize session state for storing messages (chat history)
if 'history_message' not in st.session_state:
    st.session_state.history_message = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None  # To track the last matched issue for feedback
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# Display chat history
def display_chat_history():
    if st.session_state.is_new_question:  # Display only if a new question is asked
        for message in st.session_state.history_message:
            if message.get('role') == 'user':
                with st.chat_message("user"):
                    st.markdown(f" {message['content']}")
            if message.get('role') == 'assistant':
                with st.chat_message("assistant"):
                    try:
                        for idx, response in enumerate(message['content'], 1):  # Use 'response' instead of 'message'
                            with st.expander(f"{response['Answer']} {response['Issue Type']} "):
                                st.markdown(f"{response['RCA']}")
                                st.markdown(f"{response['steps']}")
                                break
                    except:
                        pass
            
        st.session_state.is_new_question = False 
            #     st.markdown(f"**RCA:** {message['rca']}")
            #     st.markdown(f"**Steps to Resolve:** {message['steps']}")

    # st.chat_message(str(message)).markdown(str(message))
#     with st.expander(f"Previous Match {message['index']}: {message['issue_type']}"):
#         st.write(f"**RCA:** {message['rca']}\n\n"
#                  f"  **Steps to Resolve:** {message['steps']}\n\n")
#         st.write(f"**Score:** {message['score']:.2f}")

        

# User input for continuous chat
user_query = st.chat_input("Ask your IT question here...")
start_idx = (st.session_state.current_page - 1) * ANSWERS_PER_PAGE
end_idx = start_idx + ANSWERS_PER_PAGE
if user_query:
    # Store the user's message in session state
    # st.chat_message('user').markdown(user_query)
    st.session_state.history_message.append({'role': 'user', 'content': user_query})
    st.session_state.is_new_question = True
    if st.session_state.is_new_question:
        display_chat_history()
    
    # Search the FAISS vector store
    results = search_faiss_vectorstore(user_query, vectorstore, threshold=0.60, top_k=5)

    if results:
        response = []
        for idx, (result, score) in enumerate(results):
            st.session_state.history_message.append({
                'index': idx,
                'issue_type': result.metadata['Issue Type'],
                'rca': result.metadata['RCA'],
                'steps': result.metadata['Steps to Resolve'],
                'score': score
            })
            response.append({"Answer" : f"**Answer:**",
                        "Issue Type" : f"  **Matched Issue Type:** {result.metadata['Issue Type']}",
                        "RCA" : f"  **RCA:** {result.metadata['RCA']}", 
                        "steps" : f"  **Steps to Resolve:** {result.metadata['Steps to Resolve']}\n\n"})
            st.session_state.history_message.append({'role': 'assistant', 'content': response})
            with st.expander(f"Match {idx+1}: {result.metadata['Issue Type']}"):
                st.write(f"**RCA:** {result.metadata['RCA']}\n\n"
                        f"  **Steps to Resolve:** {result.metadata['Steps to Resolve']}\n\n" 
                        f"**URL:** {URL}\n\n"
                        f"**Score:** {score}")
            
                # Add button or clickable action for full details
                # Track if details are expanded
                if f'detail_shown_{idx}' not in st.session_state.expanded_results:
                    st.session_state.expanded_results[f'detail_shown_{idx}'] = False
                
                # Feedback buttons
                feedback = st.radio(f"", ('👍', '👎'), horizontal=True, key=f"feedback_{idx}",index=None)

                # Update the counters based on feedback
                # st.markdown( result.metadata['Issue Type'] result.metadata['RCA'])
                if feedback == '👍':
                    update_feedback_counter(df, result.metadata['Issue Type'], result.metadata['RCA'], 'Positive')
                    # st.success(f"Thank you for your feedback on Match {idx}!")
                elif feedback == '👎':
                    update_feedback_counter(df, result.metadata['Issue Type'], result.metadata['RCA'], 'Negative')
                    # st.warning(f"Thank you for your feedback on Match {idx}!")
  
                # Button to show or hide full details
                # if st.button(f"Show Steps to Resolve for Match {idx}", key=f"details_{idx}"):
                #     st.session_state.expanded_results[f'detail_shown_{idx}'] = True

                # # Show the "Steps to Resolve" if the button is clicked
                # if st.session_state.expanded_results[f'detail_shown_{idx}']:
                #     st.write(f"**Steps to Resolve:** {result.metadata['Steps to Resolve']}")
    
        
        # Store the assistant's response in session state
        # st.session_state.last_result = results[0][0]  # Store the top result in session state
        # Display the assistant's response
        # st.chat_message('assistant').markdown(response)
        # response = f"**Matched Issue Type:** {result['Issue Type']}\n\n**RCA:** {result['RCA']}\n\n**Steps to Resolve:** {result['Steps to Resolve']}"
        
        # # Store the assistant's response in session state
        # st.session_state.messages.append({'role': 'assistant', 'content': response})
        # st.session_state.last_result = result 



       
    else:
        # If no match is found, provide an alternate response
        st.session_state.history_message.append({'role': 'assistant', 'content': "No matching issue found. Creating a JIRA ticket..."})
        st.chat_message('assistant').markdown(f"No matching issue found. Please create ticket \n\n **URL:** {URL}")
        # Create a JIRA ticket
        # summary = f"New issue reported: {user_query}"
        # description = f"A user reported the following issue:\n\n{user_query}"
        # new_issue = create_jira_ticket(summary, description)

        # # Display the JIRA ticket details
        # ticket_message = f"JIRA Ticket Created: [{new_issue.key}]({JIRA_URL}/browse/{new_issue.key})"
        # st.session_state.messages.append({'role': 'assistant', 'content': ticket_message})
        # st.chat_message('assistant').markdown(ticket_message)