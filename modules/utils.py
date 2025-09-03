import plotly.express as px

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
