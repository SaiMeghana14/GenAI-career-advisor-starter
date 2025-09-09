import os
import tempfile
import json
import time
import math
from datetime import datetime, timedelta
from streamlit_lottie import st_lottie

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional/soft dependencies
try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False

try:
    from pyvis.network import Network
    PYVIS_OK = True
except Exception:
    PYVIS_OK = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False

try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False
    
# Configure Gemini API
import google.generativeai as genai
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-1.5-flash-002"
    model = genai.GenerativeModel(MODEL_NAME)

    # quick sanity check
    test_res = model.generate_content("ping")
    st.sidebar.success(f"✅ Gemini connected ({MODEL_NAME}): {test_res.text[:30]}...")
except Exception as e:
    st.sidebar.error(f"Gemini init failed: {e}")

# Local modules (existing features - unchanged)
from modules.resume_parser import read_pdf_text, extract_skills
from modules.recommender import (
    build_models, recommend_roles, recommend_courses,
    generate_career_roadmap, get_courses_for_career
)
from modules.utils import (
    get_api_key, bullet, plot_skill_gap, resume_feedback,
    generate_mock_interview, get_badge, generate_pdf_report, team_compatibility
)

# --------------------------
# Page & Theming
# --------------------------
st.set_page_config(page_title="GenAI Career & Skills Advisor", page_icon="🎯", layout="wide")

# Dark/Light toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
st.sidebar.toggle("🌗 Dark Mode", key="dark_mode")
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .stApp { background: #0e1117; color: #eaecef; }
        .css-1v0mbdj, .css-ffhzg2, .stMarkdown, .stText, p, li, span { color: #eaecef !important; }
        div[data-testid="stMetricValue"] { color: #eaecef !important; }
        </style>
    """, unsafe_allow_html=True)

# --------------------------
# Helper: Load Lottie file
# --------------------------
def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Lottie file: {e}")
        return None

# --------------------------
# Custom Styling (Hero, Buttons, Cards)
# --------------------------
st.markdown("""
<style>
/* Hero section */
.hero {
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2.4rem; margin-bottom: 0.5rem; }
.hero p { font-size: 1.1rem; opacity: 0.9; }

/* Card-style upload */
.upload-card {
    border: 2px dashed #ccc;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    background: #fafafa;
}
.upload-card:hover {
    border-color: #4f46e5;
    background: #f5f3ff;
}

/* Gradient button */
.stButton>button {
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    color: white;
    border-radius: 8px;
    padding: 0.7rem 1.5rem;
    font-size: 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #9333ea, #4f46e5);
}

/* Sidebar cards */
.sidebar-card {
    background: #f9fafb;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Hero Section
# --------------------------
with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown("""
        <div class="hero">
            <h1>🎯 GenAI Career & Skills Advisor</h1>
            <p>Upload your resume or type your skills to get role matches, skill gaps, roadmaps, AI feedback, mock interviews, gamified progress, and more!</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        rocket = load_lottie_file("assets/hero.json")
        if rocket:
            st_lottie(rocket, height=180, key="lottie-rocket")

# --------------------------
# Secrets Debug
# --------------------------
st.caption("🔧 Debug: Available secrets → " + str(list(st.secrets.keys())))
try:
    st.sidebar.write("🔑 GEMINI_API_KEY loaded?", "GEMINI_API_KEY" in st.secrets)
    if "GEMINI_API_KEY" in st.secrets:
        st.sidebar.success("✅ Gemini key available")
    else:
        st.sidebar.error("❌ No GEMINI_API_KEY in secrets")
except Exception as e:
    st.sidebar.error(f"Secrets error: {e}")

# --------------------------
# Data Load 
# --------------------------
careers_df = pd.read_csv("data/careers.csv")
courses_df = pd.read_csv("data/courses.csv")

# Build vectorizer/model (existing)
vec, X = build_models(careers_df)

# --------------------------
# Inputs
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📄 Resume Upload","⌨️ Type Skills", " 🤖 AI Mentor", "👥 Team Compatibility"])
user_skills = []
raw_text = ""

with tab1:
    file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            path = tmp_file.name
        raw_text = read_pdf_text(path)
        skills_found, joined = extract_skills(raw_text)
        if skills_found:
            st.success(f"Extracted skills: {', '.join(skills_found)}")
        else:
            st.info("No explicit skills detected. Try typing them in the next tab.")
        user_skills = skills_found or []

with tab2:
    typed = st.text_area("List your skills (comma separated)", placeholder="Python, SQL, IoT, ESP32, Embedded C, Git")
    if typed:
        user_skills = [s.strip() for s in typed.split(",") if s.strip()]

with tab3:
    st.subheader("Ask your AI Mentor")
    query = st.text_input("Type your question:", placeholder="e.g. What should I learn next for AI Product Manager?")
    
    if query:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(query)
        st.write("💡 Mentor says:", response.text)

with tab4:
    st.subheader("Upload teammates’ resumes")
    team_files = st.file_uploader("Upload multiple resumes", type=["pdf"], accept_multiple_files=True)

    if team_files:
        team_skills = []
        for f in team_files:
            text = read_pdf_text(f)
            skills, _ = extract_skills(text)
            team_skills.extend(skills)

        team_unique = set(team_skills)
        st.write("📌 Combined Team Skills:", ", ".join(team_unique))

        fig = go.Figure(data=[go.Bar(
            x=list(team_unique),
            y=[team_skills.count(s) for s in team_unique]
        )])
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Session defaults for trackers
if "completed_gaps" not in st.session_state:
    st.session_state.completed_gaps = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [(role/user, text)]
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {"questions": [], "answers": {}, "score": None}
    
# --------------------------
# Recommend Button
# --------------------------
if st.button("🚀 Recommend!"):
    if not user_skills:
        st.warning("Please upload a resume or type some skills first.")
        st.stop()

    st.subheader("🔎 Best-Fit Roles")
    recs = recommend_roles(user_skills, careers_df, vec, X, top_k=5)
    st.dataframe(recs[['role','summary','match_pct','have','gaps']], use_container_width=True)

    if recs.empty:
        st.error("No matching roles found. Try adding more skills.")
        st.stop()

    top_role = recs.iloc[0]
    st.markdown(f"### 🏆 Top Role: **{top_role['role']}** ({top_role['match_pct']}% match)")
    st.write(top_role['summary'])
    have = list(top_role['have']) if not isinstance(top_role['have'], list) else top_role['have']
    gaps = list(top_role['gaps']) if not isinstance(top_role['gaps'], list) else top_role['gaps']

    # Have / Need lists 
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**You already have:**")
        st.code(bullet(have) or "—")
    with colB:
        st.markdown("**You need to learn:**")
        st.code(bullet(gaps) or "No major gaps — you're ready!")

    # --------------------------
    # New: Explainable AI Insights
    # --------------------------
    st.subheader("📊 Explainable AI Insights")
    matched = set(have)
    missing = set(gaps)
    st.write(f"✅ Matched Skills: {', '.join(matched) if matched else 'None'}")
    st.write(f"❌ Missing Skills: {', '.join(missing) if missing else 'None'}")
    
    
    fig = go.Figure(data=go.Scatterpolar(r=[1 if s in matched else 0 for s in have+gaps], theta=have+gaps, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------
    # New: Gamification Progress & Badges
    # --------------------------
    progress = len(matched) / len(expected_skills)
    st.subheader("🏆 Skill Progress")
    st.progress(progress)
    
    # --------------------------
    # Personalized Career Dashboard (KPIs)
    # --------------------------
    st.subheader("📊 Career Readiness Dashboard")
    # Estimated time to career readiness (heuristic: 2 weeks per gap)
    est_weeks = max(1, math.ceil(len(gaps) * 2))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Match", f"{top_role['match_pct']}%")
    c2.metric("Skills You Have", len(have))
    c3.metric("Skills to Learn", len(gaps))
    c4.metric("ETA to Readiness", f"{est_weeks} weeks")

    # Progress bar
    total = len(have) + len(gaps) if (have or gaps) else 0
    progress_ratio = (len(have) / total) if total else 0
    st.progress(progress_ratio)

    # Optional Gauge style (using Plotly indicator)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=progress_ratio * 100,
        title={'text': "Overall Readiness %"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # --------------------------
    # Skill Gap Analyzer (existing visual using your util)
    # --------------------------
    st.subheader("📊 Skill Gap Analyzer")
    fig = plot_skill_gap(have, have + gaps)
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # Skill Progress Tracker (Gamification)
    # --------------------------
    with st.expander("🕹️ Skill Progress Tracker"):
        st.caption("Mark gaps as completed to update your readiness.")
        cols = st.columns(3)
        for i, g in enumerate(gaps):
            with cols[i % 3]:
                key = f"gap_{g}"
                checked = g in st.session_state.completed_gaps
                new_val = st.checkbox(g, value=checked, key=key)
                if new_val:
                    st.session_state.completed_gaps.add(g)
                else:
                    if g in st.session_state.completed_gaps:
                        st.session_state.completed_gaps.remove(g)

        completed_count = len(st.session_state.completed_gaps)
        remaining = [g for g in gaps if g not in st.session_state.completed_gaps]
        st.write(f"✅ Completed gaps: {completed_count} / {len(gaps)}")
        st.write("🧩 Remaining:", ", ".join(remaining) if remaining else "None — great job!")

        # Radar chart: current vs target
        skill_set = sorted(list(set(have + gaps)))
        current_vals = [1 if s in have or s in st.session_state.completed_gaps else 0 for s in skill_set]
        target_vals = [1 for _ in skill_set]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=current_vals, theta=skill_set, fill='toself', name="Current"))
        radar.add_trace(go.Scatterpolar(r=target_vals, theta=skill_set, fill='toself', name="Target"))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(radar, use_container_width=True)

    st.subheader("🛣️ Personalized Roadmap")

    roadmap = [
        ("Week 1", "Python"),
        ("Week 2", "SQL"),
        ("Week 3", "Machine Learning Basics"),
        ("Week 4", "Deep Learning"),
    ]
    
    timeline = pd.DataFrame(roadmap, columns=["Week", "Skill"])
    fig = px.timeline(timeline, x_start="Week", x_end="Week", y="Skill", color="Skill")
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("✨ Simulate Future You"):
        st.markdown("""
        ## 🔮 Your Future LinkedIn Profile (1 year later)
    
        **Name:** GenAI Student  
        **Role:** Data Scientist at TechCorp  
        **Headline:** "AI Enthusiast | Python | ML | SQL | Deep Learning"  
        **About:** Passionate about solving real-world problems using AI...
        """)

    # --------------------------
    # Interactive Visualizations
    # --------------------------
    vis_tabs = st.tabs(["☁️ Word Cloud", "🕸️ Skill → Role → Course Network", "🗺️ Roadmap Timeline"])
    with vis_tabs[0]:
        if WORDCLOUD_OK and (user_skills or have):
            wc_text = " ".join(user_skills + have)
            wc = WordCloud(width=900, height=400, background_color="white").generate(wc_text or "skills")
            import matplotlib.pyplot as plt
            fig_wc, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("Install `wordcloud` or provide skills to see a word cloud.")

    with vis_tabs[1]:
        if not PYVIS_OK:
            st.info("Install `pyvis` to see the interactive network graph.")
        else:
            net = Network(height='500px', width='100%', bgcolor='#222222' if st.session_state.dark_mode else '#ffffff',
                          font_color='white' if st.session_state.dark_mode else 'black')
            net.add_node("You", color="#6C5CE7", size=25)
            for s in have:
                net.add_node(s, color="#00B894")
                net.add_edge("You", s)
            for s in gaps:
                net.add_node(s, color="#E17055")
                net.add_edge("You", s)

            # Link to top role & some courses
            net.add_node(top_role['role'], color="#0984E3", size=20)
            net.add_edge("You", top_role['role'])
            # map some courses to gaps
            sample_courses = []
            try:
                sample_courses = recommend_courses(gaps, courses_df).head(10).to_dict("records")
            except Exception:
                pass
            for c in sample_courses:
                cname = c.get("course", "Course")
                skill = c.get("skill", None)
                net.add_node(cname, color="#FDCB6E")
                net.add_edge(top_role['role'], cname)
                if skill:
                    net.add_edge(cname, skill)

            tmp_net = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            net.show(tmp_net.name)
            with open(tmp_net.name, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=520, scrolling=True)

    with vis_tabs[2]:
        roadmap = generate_career_roadmap(top_role['role'])
        # Assign durations heuristically (2 weeks per step)
        df_timeline = []
        start = datetime.today()
        for i, step in enumerate(roadmap, 1):
            dur_weeks = 2
            s = start + timedelta(weeks=(i-1)*dur_weeks)
            e = s + timedelta(weeks=dur_weeks)
            df_timeline.append({"Task": f"Step {i}", "Description": step, "Start": s, "Finish": e})
        df_timeline = pd.DataFrame(df_timeline)
        fig_tl = px.timeline(df_timeline, x_start="Start", x_end="Finish", y="Task", color="Task", hover_data=["Description"])
        fig_tl.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_tl, use_container_width=True)

    # --------------------------
    # Courses 
    # --------------------------
    st.markdown("### 📚 Courses to Close Gaps")
    course_recs = recommend_courses(gaps, courses_df)
    if course_recs.empty:
        st.info("No gaps detected — explore advanced topics or projects.")
    else:
        st.dataframe(course_recs[['course','provider','skill','url','reason']], use_container_width=True)

    # --------------------------
    # Mock Interview Q&A (upgraded to Chat)
    # --------------------------
    st.subheader("🎤 Mock Interview Practice")
    with st.expander("Question Bank"):
        questions = generate_mock_interview(top_role['role'])
        for q in questions:
            st.markdown(f"- {q}")

    st.markdown("#### Chat with the Interviewer (AI)")
    user_msg = st.text_input("Your answer / question:")
    if st.button("Send") and user_msg:
        st.session_state.chat_history.append(("user", user_msg))
        # AI feedback via Gemini (if available)
        ai_reply = "Thanks! (Enable Gemini for richer, tailored feedback.)"
        if GEMINI_OK:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"You are a technical interviewer for the role {top_role['role']}. Provide concise, structured feedback on this candidate's answer. Answer:\n{user_msg}"
                res = model.generate_content(prompt)
                ai_reply = (res.text or "").strip() or ai_reply
            except Exception as e:
                ai_reply = f"(AI feedback error: {e})"
        st.session_state.chat_history.append(("ai", ai_reply))
        st.toast("Interviewer responded!")

    if st.session_state.chat_history:
        for who, text in st.session_state.chat_history[-12:]:
            if who == "user":
                st.chat_message("user").write(text)
            else:
                st.chat_message("assistant").write(text)

    if WEBRTC_OK:
        with st.expander("🎙️ Optional: Practice with Voice (beta)"):
            st.caption("Record your answer; transcript/analysis not stored.")
            webrtc_streamer(key="voice", mode=WebRtcMode.SENDONLY, audio_receiver_size=1024)

    # --------------------------
    # Role Comparison
    # --------------------------
    st.subheader("📍 Career Role Comparison")
    roles_to_compare = st.multiselect("Select up to 3 roles to compare", options=recs['role'].tolist(), default=[top_role['role']])
    if roles_to_compare:
        comp = careers_df[careers_df['role'].isin(roles_to_compare)].copy()
        show_cols = [c for c in ["role","summary","avg_salary","growth_outlook","skills"] if c in comp.columns]
        if show_cols:
            st.dataframe(comp[show_cols], use_container_width=True)
        # Radar on skill coverage union
        union_skills = sorted(list(set().union(*[
            (set(eval(x)) if isinstance(x, str) and x.startswith("[") else set(x))
            if 'skills' in comp.columns else set()
            for x in comp['skills']]))) if ('skills' in comp.columns and not comp.empty) else sorted(list(set(have+gaps)))
        if union_skills:
            rfig = go.Figure()
            for _, row in comp.iterrows():
                rs = row.get('skills', [])
                if isinstance(rs, str) and rs.startswith("["):
                    try: rs = list(eval(rs))
                    except Exception: rs = []
                vals = [1 if s in (rs or []) else 0 for s in union_skills]
                rfig.add_trace(go.Scatterpolar(r=vals, theta=union_skills, fill='toself', name=row['role']))
            rfig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
            st.plotly_chart(rfig, use_container_width=True)

    # --------------------------
    # Market Insights
    # --------------------------
    st.subheader("🌍 Market Insights")
    st.caption("Upload a CSV with columns like: region, skill, demand (0-100).")
    mi_file = st.file_uploader("Upload job trends CSV", type=["csv"], key="mi")
    if mi_file:
        try:
            mi_df = pd.read_csv(mi_file)
            # Heatmap: skills vs region
            if {"region","skill","demand"}.issubset(set(mi_df.columns)):
                heat = mi_df.pivot_table(index="skill", columns="region", values="demand", aggfunc="mean").fillna(0)
                fig_hm = px.imshow(heat, aspect="auto", title="In-demand Skills by Region (Heatmap)")
                st.plotly_chart(fig_hm, use_container_width=True)
                # Optional choropleth if region is country code
                if "country_code" in mi_df.columns:
                    ch = px.choropleth(mi_df, locations="country_code", color="demand",
                                       hover_name="region", color_continuous_scale="Blues", title="Demand by Country")
                    st.plotly_chart(ch, use_container_width=True)
            else:
                st.info("CSV missing required columns: region, skill, demand")
        except Exception as e:
            st.error(f"Failed to parse Market Insights CSV: {e}")
    else:
        st.caption("No CSV uploaded. Provide one to view heatmaps/choropleths.")

    # --------------------------
    # Career Quiz Mode (Gemini MCQs)
    # --------------------------
    st.subheader("💡 Career Quiz Mode")
    st.caption("Get 5 MCQs for the selected top role. Scores locally, no data stored.")
    def gen_quiz(role):
        if not GEMINI_OK:
            return [{"q":"Gemini not available. Install & set key.", "opts":["A","B","C","D"], "ans":"A"}]
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Create 5 multiple-choice questions (MCQ) for the role '{role}'.
        Return strictly JSON list with fields: q (string), opts (list of 4 options), ans (exact option text).
        Example:
        [{{"q":"...","opts":["opt1","opt2","opt3","opt4"],"ans":"opt2"}}, ...]
        """
        res = model.generate_content(prompt)
        text = res.text
        try:
            data = json.loads(text)
            return data
        except Exception:
            # try to extract json substring
            start = text.find("[")
            end = text.rfind("]")+1
            return json.loads(text[start:end])

    if st.button("Generate Quiz"):
        try:
            st.session_state.quiz_state["questions"] = gen_quiz(top_role['role'])
            st.session_state.quiz_state["answers"] = {}
            st.session_state.quiz_state["score"] = None
        except Exception as e:
            st.error(f"Quiz generation error: {e}")

    qs = st.session_state.quiz_state["questions"]
    if qs:
        for i, q in enumerate(qs):
            st.markdown(f"**Q{i+1}. {q['q']}**")
            key = f"quiz_{i}"
            st.session_state.quiz_state["answers"][i] = st.radio(
                "Choose one:", q["opts"], key=key, horizontal=True)
        if st.button("Submit Quiz"):
            score = 0
            for i, q in enumerate(qs):
                if st.session_state.quiz_state["answers"].get(i) == q["ans"]:
                    score += 1
            st.session_state.quiz_state["score"] = score
            st.success(f"Your score: {score}/{len(qs)}")
            if score == len(qs): st.balloons()
                
    st.subheader("📌 Latest Jobs (Mock Integration)")
    jobs = [
        {"role": "Data Scientist", "company": "Google", "location": "Bangalore"},
        {"role": "AI Engineer", "company": "Microsoft", "location": "Hyderabad"},
    ]
    for job in jobs:
        st.write(f"**{job['role']}** at {job['company']} ({job['location']})")

    # --------------------------
    # Portfolio Integration
    # --------------------------
    st.subheader("📥 Portfolio Integration")
    gh_user = st.text_input("GitHub username (optional):")
    if gh_user and REQUESTS_OK:
        try:
            r = requests.get(f"https://api.github.com/users/{gh_user}/repos?per_page=100", timeout=10)
            repos = r.json() if r.ok else []
            if isinstance(repos, list) and repos:
                df_repos = pd.DataFrame([{
                    "name": x.get("name"),
                    "stars": x.get("stargazers_count"),
                    "language": x.get("language"),
                    "updated": x.get("updated_at"),
                    "url": x.get("html_url")
                } for x in repos])
                df_repos = df_repos.sort_values("stars", ascending=False)
                st.dataframe(df_repos.head(10), use_container_width=True)
            else:
                st.info("No public repos found or API rate-limited.")
        except Exception as e:
            st.error(f"GitHub fetch error: {e}")
    elif gh_user and not REQUESTS_OK:
        st.info("Install `requests` to enable GitHub integration.")

    # --------------------------
    # Resume Feedback 
    # --------------------------
    st.subheader("💡 Resume Feedback")
    try:
        feedback = resume_feedback(user_skills, top_role['role'])
        st.write(feedback)
    except ValueError as e:
        st.error(str(e))

    # --------------------------
    # Gamification Badge 
    # --------------------------
    st.subheader("🏅 Your Skill Badge")
    badge = get_badge(user_skills)
    st.success(f"Your Badge: {badge}")

    # Badges
    if len(matched) >= 3:
        st.success("🔥 You unlocked the **AI Explorer Badge**!")
    if len(matched) >= 5:
        st.success("🚀 You unlocked the **Data Pro Badge**!")
    
    # --------------------------
    # Export Career Report 
    # --------------------------
    st.subheader("📥 Download Personalized Career Report")
    if st.button("Generate PDF Report"):
        filename = "career_report.pdf"
        courses = course_recs.to_dict(orient="records") if not course_recs.empty else []
        generate_pdf_report(filename, user_skills, top_role['role'], roadmap, courses)
        with open(filename, "rb") as f:
            st.download_button("Download PDF", f, file_name="career_report.pdf")

    # --------------------------
    # Team Collaboration (upgraded)
    # --------------------------
    st.subheader("🤝 Team Collaboration Mode")
    team_input = st.text_area("Enter skills of your teammates (comma-separated per line)")
    if team_input:
        team_skills = [set(s.strip() for s in line.split(",") if s.strip()) for line in team_input.splitlines()]
        score, combined = team_compatibility(team_skills)
        st.write(f"**Team Compatibility Score:** {score}")
        st.write(f"**Combined Skills:** {', '.join(combined)}")

        # Team radar chart (frequency of skills)
        freq = {}
        for sset in team_skills:
            for s in sset:
                freq[s] = freq.get(s, 0) + 1
        labels = list(freq.keys())[:20]  # limit for readability
        values = [freq[k] for k in labels]
        tr = go.Figure()
        tr.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', name="Team"))
        tr.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(tr, use_container_width=True)

        # Simple role suggestions based on distribution
        suggestions = []
        if any(x in combined for x in ["Figma","UI","Design","UX"]): suggestions.append("Designer")
        if any(x in combined for x in ["Python","C++","JavaScript","SQL","Data Structures"]): suggestions.append("Developer")
        if any(x in combined for x in ["Analysis","Excel","PowerBI","Tableau","SQL","Pandas"]): suggestions.append("Analyst")
        if any(x in combined for x in ["Leadership","Scrum","Agile","Roadmap","Communication"]): suggestions.append("Leader")
        st.info("Suggested Team Roles: " + (", ".join(sorted(set(suggestions))) or "Add more data to get suggestions"))

else:
    st.info("Upload a resume or type your skills, then click **Recommend!**")

# --------------------------
# Sidebar Dashboard
# --------------------------
st.sidebar.markdown("## 📊 Career Dashboard")

# If user has results, show KPIs
if "recs" in locals() and not recs.empty:
    top_role = recs.iloc[0]
    match_pct = int(top_role['match_pct'])
    have_count = len(top_role['have']) if isinstance(top_role['have'], (list,set)) else 0
    gaps_count = len(top_role['gaps']) if isinstance(top_role['gaps'], (list,set)) else 0
    readiness = round(match_pct * (have_count / (have_count + gaps_count + 1)), 1)

    with st.sidebar.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.metric("🎯 Match %", f"{match_pct}%")
        st.metric("📚 Skills You Have", have_count)
        st.metric("🧩 Skills to Learn", gaps_count)
        st.metric("⏳ Readiness Score", f"{readiness}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Progress bar
        st.progress(min(1.0, match_pct/100))

# Gemini API Key status
st.sidebar.markdown("## 🔑 API Key")
if "GEMINI_API_KEY" in st.secrets:
    st.sidebar.success("Gemini key available")
else:
    st.sidebar.error("No GEMINI_API_KEY found")

# About / Features
st.sidebar.markdown("## ℹ️ About")
st.sidebar.write("Built for **GenAI Exchange Hackathon** 🚀")
st.sidebar.write("Features: Resume Parsing, Role Matching, Skill Gap Charts, Roadmaps, AI Feedback, Mock Interviews, Gamification, Team Mode, PDF Reports.")

st.sidebar.markdown("## ✨ Upgrades")
st.sidebar.write("""
- KPI Dashboard  
- Progress Tracker  
- Word Cloud  
- Network Graph  
- Timeline Roadmap  
- Interview Chat  
- Role Comparison  
- Market Insights  
- Quiz Mode  
- Portfolio View  
- Team Radar  
- Lottie Animations  
- Dark/Light Mode  
- Toasts & Balloons  
""")

st.sidebar.info("MIT License")

