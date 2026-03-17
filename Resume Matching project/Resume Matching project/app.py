import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("Resume Matcher")
st.write("Match your profile against the Job Bank")

# Input
resume_text = st.text_area("Paste your Resume here")
df = pd.read_csv('/content/drive/MyDrive/Resume Matching project/Data/job_dataset.csv') 

if st.button("Find Best Jobs"):
    if resume_text:
        df['context'] = df['Title'] + " " + df['Skills'] + " " + df['Responsibilities']
        job_embs = model.encode(df['context'].tolist(), convert_to_tensor=True)
        res_emb = model.encode(resume_text, convert_to_tensor=True)
        
        scores = util.cos_sim(res_emb, job_embs)[0].cpu().numpy() * 100
        df['Score'] = scores
        
        top_matches = df.sort_values(by='Score', ascending=False).head(5)
        
        st.subheader("Top 5 Job Matches:")
        for i, row in top_matches.iterrows():
            st.write(f"**{row['Title']}** - Match: {row['Score']:.2f}%")
            st.progress(int(row['Score']))
    else:
        st.warning("Please paste your resume!")
