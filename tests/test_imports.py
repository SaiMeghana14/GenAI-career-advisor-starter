def test_imports():
    try:
        import streamlit
        import utils.career_mapping
        import utils.recommendations
        import utils.resume_parser
    except Exception as e:
        assert False, f"Import failed: {e}"
