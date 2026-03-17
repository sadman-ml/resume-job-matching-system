# **PROJECT OVERVIEW**
I created an intelligent Machine Learning system that will automate the hiring process by matching resumes with the most relevant job descriptions. This project is not only about matching keywords in resumes and job descriptions but also about understanding the context and intention of technical resumes.

### **Project Aim:**
1. Data load and clean from job dataset.
2. Found feature from text using TF-IDF vectorization.
3. Finding out job similarities with resume using Cosine Similarity.
4. Showed Top 10 matching jobs.
5. Created visualizations.

## Visualizations & Results
The system provides clear insights through:
- **Match Score Distribution:** Understanding how well the candidate pool aligns with a job.
- **Top 10 Rankings:** Instantly identifying the most qualified candidates.
- **Correlation Heatmaps:** Analyzing the relationship between experience and match scores.

## Tech Stack
- **Language:** Python
- **AI/NLP:** Sentence-Transformers (BERT), Scikit-Learn
- **Data Engineering:** Pandas, NumPy
- **Web UI:** Streamlit
- **Visualization:** Matplotlib, Seaborn

## Repository Structure
- `resume_matching.ipynb`: Complete research, cleaning, and model development.
- `app.py`: Streamlit source code for the interactive web application.
- `job_dataset.csv`: The primary dataset containing job roles and requirements.
- `figures/`: Directory containing generated analytical plots.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn streamlit sentence-transformers`.
3. Run the dashboard: `streamlit run app.py`.

## Why this project?
I developed this to explore the intersection of **Artificial Intelligence** and **Human Resources**. By leveraging Transformer-based models like **all-MiniLM-L6-v2**, this project demonstrates a modern approach to solving real-world data science problems.

---
⭐ **If you find this project helpful, please give it a star!**

**Developed by: Sadman Ahmed**
