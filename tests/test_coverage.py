from scad_rag.features.coverage import keyword_coverage


def test_keyword_coverage():
    assert keyword_coverage("Python was created by Guido.", "Python was created by Guido in 1991.") == 1.0
