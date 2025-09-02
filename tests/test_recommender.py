import pandas as pd
from modules.recommender import build_models, recommend_roles

def test_recommend_roles_basic():
    careers_df = pd.DataFrame({
        'role': ['IoT Engineer','Data Analyst'],
        'summary': ['Work on IoT','Work on data'],
        'required_skills': ['IoT;ESP32;Arduino','SQL;Python;Power BI']
    })
    vec, X = build_models(careers_df)
    user_skills = ['IoT','ESP32']
    out = recommend_roles(user_skills, careers_df, vec, X, top_k=1)
    assert out.iloc[0]['role'] == 'IoT Engineer'
