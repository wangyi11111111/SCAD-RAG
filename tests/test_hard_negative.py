from scad_rag.features.hard_negative import select_hard_negative
from scad_rag.models.dummy_models import DummyEmbedder, DummyNLIModel
from scad_rag.schema import Evidence


def test_hard_negative_prefers_marked_candidate():
    ev, _ = select_hard_negative(
        "Apollo 11 landed on the Moon in 1969.",
        "e1",
        [
            Evidence("e1", "Apollo 11 landed on the Moon in 1969.", "gold"),
            Evidence("e2", "Apollo 13 launched in 1970 but did not land on the Moon.", "hard_negative"),
        ],
        DummyEmbedder(),
        DummyNLIModel(),
        {"entailment_threshold": 0.5},
    )
    assert ev is not None
    assert ev.id == "e2"
