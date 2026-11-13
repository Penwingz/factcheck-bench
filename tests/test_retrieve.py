from factcheck.retrieve import Index, title_of, tokenize

POOL = [
    {"doc_id": "Eiffel_Tower", "sentence_id": 0,
     "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris , France ."},
    {"doc_id": "Berlin", "sentence_id": 0, "text": "Berlin is the capital of Germany ."},
    {"doc_id": "Jackie_-LRB-2016_film-RRB-", "sentence_id": 0,
     "text": "It was directed by Pablo Larrain ."},
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Eiffel Tower is in Berlin") == ["eiffel", "tower", "berlin"]


def test_title_unescaped():
    assert title_of("Jackie_-LRB-2016_film-RRB-") == "Jackie (2016 film)"


def test_top_k_ranks_and_uses_title():
    idx = Index(POOL)
    hits = idx.top_k("Where is the Eiffel Tower?", k=2)
    assert hits[0].doc_id == "Eiffel_Tower" and len(hits) == 2
    # the pronoun sentence is found through its page title
    assert idx.top_k("Jackie film director", k=1)[0].doc_id.startswith("Jackie")
    assert hits[0].text.startswith("The Eiffel Tower is")  # title already present
    assert idx.items[2].text == "Jackie (2016 film): It was directed by Pablo Larrain."
