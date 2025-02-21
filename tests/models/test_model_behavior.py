import pytest


@pytest.mark.parametrize(
    "example_a, example_b, label",
    [
        (
            "This week, the Texas Court of Appeals made a ruling that is both outrageous and grotesque",
            "This week, the Texas Court of Appeals declared a ruling that is both outrageous and grotesque"
            "NEWS & POLITICS",
        ),
    ],
)
def test_model_invariance(example_a, example_b, model):
    pass


@pytest.mark.parametrize(
    "example, label",
    [
        (
            "Like most children, my 5-year-old is obsessed with Christmas. She loves brightly colored lights." ,
            "PARENTING",
        ),
        (
            "What 4-and-a-half-year-old girl wouldn't want to spend a day with Elsa from Frozen?",
            "PARENTING",
        ),
        (
            "what better way to stretch it out than with a few yoga poses? This quick-hit routine is designed to make you feel alert.",
            "SPORTS & WELLNESS",
        ),
    ],
)
def test_model_direction(example, label):
    pass

