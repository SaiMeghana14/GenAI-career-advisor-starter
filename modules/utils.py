import plotly.express as px
import google.generativeai as genai
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Skill Gap Analyzer
def plot_skill_gap(user_skills, required_skills):
    """Visualize which skills are present vs missing"""
    data = []
    for skill in required_skills:
        data.append({
            "Skill": skill,
            "Status": "Have" if skill in user_skills else "Missing"
        })

    fig = px.bar(
        data,
        x="Skill",
        color="Status",
        title="Skill Gap Analysis",
        barmode="group"
    )
    return fig

# Resume Feedback using Gemini
def resume_feedback(user_skills, target_career):
    prompt = f"My resume shows skills: {user_skills}. I want to become a {target_career}. What am I missing?"
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "AI feedback not available (missing API key)."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI feedback not available: {e}"

# Mock Interview Q&A
def generate_mock_interview(career):
    questions = {
        "Data Scientist": [
            "What is the difference between supervised and unsupervised learning?",
            "Explain the bias-variance tradeoff."
        ],
        "Embedded Engineer": [
            "Explain how an ESP32 differs from Arduino Uno.",
            "What is the role of RTOS in embedded systems?"
        ]
    }
    return questions.get(career, ["Tell me about yourself."])

# Gamification Badge
def get_badge(user_skills):
    if len(user_skills) > 8:
        return "🏆 AI Master"
    elif len(user_skills) > 5:
        return "🥈 Skilled Learner"
    else:
        return "🎯 Beginner Explorer"

# PDF Career Report
def generate_pdf_report(filename, skills, career, roadmap, courses):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Career Advisor Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Target Career: {career}", styles['Heading2']))

    elements.append(Paragraph("Your Skills:", styles['Heading3']))
    elements.append(Paragraph(", ".join(skills), styles['Normal']))

    elements.append(Paragraph("Roadmap:", styles['Heading3']))
    for i, step in enumerate(roadmap, 1):
        elements.append(Paragraph(f"{i}. {step}", styles['Normal']))

    elements.append(Paragraph("Courses:", styles['Heading3']))
    for c in courses:
        elements.append(Paragraph(f"{c['course']} ({c['provider']}) - {c['url']}", styles['Normal']))

    doc.build(elements)
    return filename

# Team Collaboration
def team_compatibility(team_skills):
    all_skills = set().union(*team_skills)
    return len(all_skills), all_skills
