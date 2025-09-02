from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _prep_req_skills(s: str) -> List[str]:
    return [x.strip() for x in s.split(';') if x.strip()]

def build_models(careers_df: pd.DataFrame):
    corpus = careers_df['required_skills'].fillna('')
    vec = TfidfVectorizer(token_pattern=r'[A-Za-z+#.]+')
    X = vec.fit_transform(corpus)
    return vec, X

def recommend_roles(user_skills: List[str], careers_df: pd.DataFrame, vec, X, top_k: int = 5) -> pd.DataFrame:
    query = " ".join(user_skills)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    out = careers_df.copy()
    out['similarity'] = sims
    out = out.sort_values('similarity', ascending=False).head(top_k).reset_index(drop=True)
    out['required_list'] = out['required_skills'].apply(_prep_req_skills)
    user_set = set([s.lower() for s in user_skills])
    out['have'] = out['required_list'].apply(lambda req: [s for s in req if s.lower() in user_set])
    out['gaps'] = out['required_list'].apply(lambda req: [s for s in req if s.lower() not in user_set])
    out['match_pct'] = out.apply(lambda r: int(100*len(r['have'])/max(1,len(r['required_list']))), axis=1)
    return out[['role','summary','match_pct','have','gaps','required_skills','similarity']]

def recommend_courses(gap_skills: List[str], courses_df: pd.DataFrame) -> pd.DataFrame:
    if not gap_skills:
        return pd.DataFrame(columns=courses_df.columns.tolist()+['reason'])
    skills_lower = [g.lower() for g in gap_skills]
    mask = courses_df['skill'].str.lower().isin(skills_lower)
    out = courses_df[mask].copy()
    out['reason'] = out['skill'].apply(lambda s: f"To close the '{s}' gap")
    return out
