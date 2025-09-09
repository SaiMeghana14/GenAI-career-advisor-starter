## GenAI Career Advisor 🎓🤖

AI-powered career guidance platform for personalized skill-building, mentorship, and team collaboration.

[👉 Live Demo](https://genai-career-advisor-starter-ladamzyucowm4jmo2wx3jh.streamlit.app/)

---

## 💡 Problem Statement

Career guidance today is often generic, inaccessible, or outdated. Students and professionals struggle to:

Identify career paths that match their skills.

Access personalized mentorship.

Find the right courses to upskill.

Build effective teams based on complementary skills.

---

## 🚀 Our Solution

GenAI Career Advisor leverages Google Gemini AI + curated datasets to provide:

📄 Resume Parsing – Extract skills directly from resumes.

🤖 AI Mentor – Personalized career guidance with GenAI.

📊 Recommendations Engine – Suggests courses, skills, and career paths.

👥 Team Compatibility – Smart matching for hackathon/project teams.

🌍 Inclusive Guidance – Tailored for both students and professionals.

---

## 🔥 Key Highlights (Why It’s Innovative)

Uses state-of-the-art LLMs (Gemini) for mentorship.

Gamified skill-building with career roadmaps.

Lightweight, deployable Streamlit app → runs anywhere.

Team compatibility adds collaboration potential beyond personal guidance.

---

## 📂 Project Structure
```
├── .github/workflows/ci.yml        # CI/CD pipeline
├── assets/
│   └── hero.json                   # App assets (UI/branding)
├── data/
│   ├── careers.csv                  # Career paths dataset
│   ├── courses.csv                  # Courses dataset
│   └── skills.csv                   # Skills dataset
├── modules/
│   ├── recommender.py               # Recommendation engine
│   ├── resume_parser.py             # Resume parsing logic
│   └── utils.py                     # Helper utilities
├── tests/
│   ├── test_imports.py              # Import checks
│   └── test_recommender.py          # Unit tests for recommender
├── app.py                           # Main Streamlit app
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```
---

## 🛠️ Tech Stack

Frontend: Streamlit

Backend/Logic: Python 3.13

AI: Google Generative AI (Gemini 1.5 Pro/Flash)

Data: Skills, Careers & Courses CSVs

Infra: GitHub Actions, Streamlit Cloud

---

## ⚙️ Installation & Run
```
git clone https://github.com/your-username/genai-career-advisor.git
cd genai-career-advisor
pip install -r requirements.txt
streamlit run app.py
```
---

## 🔑 API Setup (Google Gemini)

Enable Generative Language API in Google AI Studio
.

Create an API key.

Add to environment:

export GOOGLE_API_KEY="your-api-key"


(For Streamlit Cloud → add under Secrets).

---

## 🧪 Testing
pytest tests/

---

## 📊 Scalability & Real-World Impact

Scalable: Can integrate with job portals, edtech platforms, or LMS.

Accessible: Runs in the browser with minimal infra requirements.

Real-world use cases:

University career cells.

Online learning platforms.

Hackathons and team-matching platforms.

---

## 🌱 Future Scope

AI-powered career roadmap visualization.

Integration with LinkedIn & job APIs.

Multilingual support for global inclusivity.

Real-time collaboration rooms for team matching.

---

## 👩‍💻 Team

Team Name: Your Hackathon Team Name

🧑‍💻 Developer 1 – Resume parsing & recommender engine

🤖 Developer 2 – AI Mentor integration

🎨 Developer 3 – UI/UX & Streamlit workflows

🔬 Developer 4 – Data handling & testing

---

## 🌟 Expected Impact

By blending AI-driven mentorship + skill matching + team compatibility, GenAI Career Advisor empowers learners and professionals to make smarter career decisions, upskill effectively, and thrive in collaborative environments.
