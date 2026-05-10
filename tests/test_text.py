from scad_rag.utils.text import split_sentences, tokenize


def test_sentence_splitting():
    assert split_sentences("A is true. B is false.") == ["A is true.", "B is false."]


def test_tokenize_stopwords():
    tokens = tokenize("The model is a Transformer.")
    assert "the" not in tokens
    assert "transformer" in tokens
