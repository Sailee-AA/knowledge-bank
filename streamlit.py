import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Streamlit app title
st.title('Incident Resolver Chatbot')


# Input box for user question
user_query = st.text_input('Please enter your query:', '')

# Load CSV data into pandas
data = pd.read_csv('chatbot_merged_data.csv')

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to find the closest matching issue type
def find_closest_issue(query, issues):
    query_embedding = model.encode(query, convert_to_tensor=True)
    issue_embeddings = model.encode(issues, convert_to_tensor=True)
    
    # Find the closest issue based on cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, issue_embeddings)
    closest_index = cosine_scores.argmax()
    return closest_index, cosine_scores[0][closest_index]



if st.button('Generate Response'):
    if user_query:
        # Extract issue types for comparison
        issue_types = data['Issue Type'].tolist()

        

        # Find the closest matching issue type
        closest_idx, score = find_closest_issue(user_query, issue_types)
        # st.write((closest_idx,score))

        # Convert closest_idx to an integer
        closest_idx = closest_idx.item()  # Extract the integer from the tensor
        
        if score > 0.5:  # If a close enough match is found
            issue_found = data.iloc[closest_idx]
            st.write(f"**Matched Issue Type:** {issue_found['Issue Type']}")
            st.write(f"**RCA:** {issue_found['RCA']}")
            st.write(f"**Steps to Resolve:** {issue_found['Steps to Resolve']}")

            # Thumbs up/thumbs down for feedback
            feedback = st.radio("Was this helpful?", ('👍', '👎'))

            # Update the counters based on feedback
            if feedback == '👍':
                data.loc[closest_idx, 'Positive_Counter'] += 1
            elif feedback == '👎':
                data.loc[closest_idx, 'Negative_Counter'] += 1
        else:
            st.write("No matching issue found. Raising a ticket...")
            
    else:
        st.write("Please enter a query.")

