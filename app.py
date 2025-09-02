import streamlit as st
import pandas as pd
from modules.resume_parser import read_pdf_text, extract_skills
from modules.recommender import build_models, recommend_roles, recommend_courses
from modules.utils import get_api_key, bullet

st.set_page_config(page_title="GenAI Career & Skills Advisor", page_icon="🎯", layout="wide")

st.title("🎯 GenAI Career & Skills Advisor")
st.write("Upload your resume or type your skills to get role matches, skill gaps, courses, and a roadmap.")

# Load data
careers_df = pd.read_csv("data/careers.csv")
courses_df = pd.read_csv("data/courses.csv")

# Build vectorizer/model
vec, X = build_models(careers_df)

tab1, tab2 = st.tabs(["Resume Upload","Type Skills"])

user_skills = []
raw_text = ""

with tab1:
    file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if file is not None:
        path = f"/mnt/data/_resume_tmp.pdf"
        with open(path,"wb") as f:
            f.write(file.read())
        raw_text = read_pdf_text(path)
        skills_found, joined = extract_skills(raw_text)
        st.success(f"Extracted skills: {', '.join(skills_found) if skills_found else 'None'}")
        user_skills = skills_found

with tab2:
    typed = st.text_area("List your skills (comma separated)", placeholder="Python, SQL, IoT, ESP32, Embedded C, Git")
    if typed:
        user_skills = [s.strip() for s in typed.split(",") if s.strip()]

st.divider()

if st.button("Recommend!"):
    if not user_skills:
        st.warning("Please upload a resume or type some skills first.")
        st.stop()

    st.subheader("🔎 Best-Fit Roles")
    recs = recommend_roles(user_skills, careers_df, vec, X, top_k=5)
    st.dataframe(recs[['role','summary','match_pct','have','gaps']])

    top_role = recs.iloc[0]
    st.markdown(f"### 🏆 Top Role: **{top_role['role']}** ({top_role['match_pct']}% match)")
    st.write(top_role['summary'])
    have = top_role['have']
    gaps = top_role['gaps']

    st.markdown("**You already have:**")
    st.code(bullet(have) or "—")

    st.markdown("**You need to learn:**")
    st.code(bullet(gaps) or "No major gaps — you're ready!")

    st.markdown("### 📚 Courses to Close Gaps")
    course_recs = recommend_courses(gaps, courses_df)
    if course_recs.empty:
        st.info("No gaps detected — explore advanced topics or projects.")
    else:
        st.dataframe(course_recs[['course','provider','skill','url','reason']])

    st.markdown("### 🗺️ 8-Week Roadmap")
    if gaps:
        steps = [
            "Weeks 1–2: Master the basics of " + (gaps[0] if len(gaps)>0 else ""),
            "Weeks 3–4: Build a mini-project applying the above skill",
            "Weeks 5–6: Learn " + (gaps[1] if len(gaps)>1 else "an advanced concept in your chosen role"),
            "Weeks 7–8: Capstone project + polish portfolio and resume",
        ]
    else:
        steps = [
            "Weeks 1–2: Build a portfolio project in your target role",
            "Weeks 3–4: Add tests, docs, and deploy your project",
            "Weeks 5–6: Prepare for interviews (DSA + system/design as relevant)",
            "Weeks 7–8: Network, apply to 10–15 positions, refine applications",
        ]
    st.code(bullet(steps))

    key_name, api_key = get_api_key()
    if api_key:
        st.caption(f"LLM provider detected: {key_name}. (Hook up your advice module here.)")
    else:
        st.caption("Tip: Add an API key in .env to generate AI-tailored narrative advice.")

else:
    st.info("Upload a resume or type your skills, then click **Recommend!**")

st.sidebar.header("About")
st.sidebar.write("Built for GenAI Exchange Hackathon. Replace `data/*.csv` with richer datasets.")
st.sidebar.write("MIT License")
