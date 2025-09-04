import streamlit as st
import pandas as pd
from modules.resume_parser import read_pdf_text, extract_skills
from modules.recommender import build_models, recommend_roles, recommend_courses, generate_career_roadmap, get_courses_for_career
from modules.utils import get_api_key, bullet, plot_skill_gap, resume_feedback, generate_mock_interview, get_badge, generate_pdf_report, team_compatibility

st.set_page_config(page_title="GenAI Career & Skills Advisor", page_icon="🎯", layout="wide")

st.title("🎯 GenAI Career & Skills Advisor")
st.write("Upload your resume or type your skills to get role matches, skill gaps, courses, roadmap, AI feedback, and more!")

# Load data
careers_df = pd.read_csv("data/careers.csv")
courses_df = pd.read_csv("data/courses.csv")

# Build vectorizer/model
vec, X = build_models(careers_df)

tab1, tab2 = st.tabs(["Resume Upload","Type Skills"])

user_skills = []
raw_text = ""

import tempfile

with tab1:
    file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if file is not None:
        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            path = tmp_file.name

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
    
    if recs.empty:
        st.error("No matching roles found. Try adding more skills.")
    st.stop()

    top_role = recs.iloc[0]
    st.markdown(f"### 🏆 Top Role: **{top_role['role']}** ({top_role['match_pct']}% match)")
    st.write(top_role['summary'])
    have = top_role['have']
    gaps = top_role['gaps']

    st.markdown("**You already have:**")
    st.code(bullet(have) or "—")

    st.markdown("**You need to learn:**")
    st.code(bullet(gaps) or "No major gaps — you're ready!")

    # ✅ Skill Gap Analyzer (Visual)
    st.subheader("📊 Skill Gap Analyzer")
    fig = plot_skill_gap(have, have + gaps)
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Course Recs
    st.markdown("### 📚 Courses to Close Gaps")
    course_recs = recommend_courses(gaps, courses_df)
    if course_recs.empty:
        st.info("No gaps detected — explore advanced topics or projects.")
    else:
        st.dataframe(course_recs[['course','provider','skill','url','reason']])

    # ✅ Career Roadmap
    st.markdown("### 🛤 Career Roadmap")
    roadmap = generate_career_roadmap(top_role['role'])
    for i, step in enumerate(roadmap, 1):
        st.markdown(f"**Step {i}:** {step}")

    # ✅ Resume Feedback (AI-powered)
    st.subheader("💡 Resume Feedback")
    key_name, api_key = get_api_key()
    if api_key:
        feedback = resume_feedback(user_skills, top_role['role'])
        st.write(feedback)
    else:
        st.info("Add an API key in `.env` to enable AI-powered resume feedback.")

    # ✅ Mock Interview Q&A
    st.subheader("🎤 Mock Interview Practice")
    questions = generate_mock_interview(top_role['role'])
    for q in questions:
        st.markdown(f"- {q}")

    # ✅ Gamification Badge
    st.subheader("🏅 Your Skill Badge")
    badge = get_badge(user_skills)
    st.success(f"Your Badge: {badge}")

    # ✅ Export Career Report
    st.subheader("📥 Download Personalized Career Report")
    if st.button("Generate PDF Report"):
        filename = "career_report.pdf"
        generate_pdf_report(filename, user_skills, top_role['role'], roadmap, course_recs.to_dict(orient="records"))
        with open(filename, "rb") as f:
            st.download_button("Download PDF", f, file_name="career_report.pdf")

    # ✅ Team Collaboration Mode
    st.subheader("🤝 Team Collaboration Mode")
    team_input = st.text_area("Enter skills of your teammates (comma-separated per line)")
    if team_input:
        team_skills = [set(s.strip() for s in line.split(",") if s.strip()) for line in team_input.splitlines()]
        score, combined = team_compatibility(team_skills)
        st.write(f"**Team Compatibility Score:** {score}")
        st.write(f"**Combined Skills:** {', '.join(combined)}")

else:
    st.info("Upload a resume or type your skills, then click **Recommend!**")

st.sidebar.header("About")
st.sidebar.write("Built for GenAI Exchange Hackathon 🚀")
st.sidebar.write("Features: Resume Parsing, Role Matching, Skill Gap Charts, Roadmaps, AI Feedback, Mock Interviews, Gamification, Team Mode, PDF Reports.")
st.sidebar.write("MIT License")
