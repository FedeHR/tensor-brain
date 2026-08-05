import torch
from torch.nn import functional as F

from experiments.pvsg.baselines import PredicateComplementarity, PredicatePriors


def test_predicate_priors_fit_frequency_and_directed_category_pairs() -> None:
    priors = PredicatePriors.fit(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        ("person", "person", "dog"),
        ("ball", "ball", "person"),
    )

    assert priors.frequency_logits.argmax().item() == 0
    predictions = priors.logits(("person", "unknown"), ("ball", "unknown"))
    assert predictions[0].argmax().item() == 0
    torch.testing.assert_close(predictions[1], priors.frequency_logits)


def test_predicate_priors_fit_symbolic_records_without_dense_targets() -> None:
    priors = PredicatePriors.fit_records(
        (
            {
                "subject_category": "person",
                "object_category": "ball",
                "predicates": ("holding",),
            },
            {
                "subject_category": "dog",
                "object_category": "person",
                "predicates": ("looking at", "unknown"),
            },
        ),
        ("holding", "looking at"),
    )

    assert priors.frequency_logits.argmax().item() == 0
    assert priors.logits(("dog",), ("person",)).argmax().item() == 1


def test_predicate_priors_materialize_frequency_fallback() -> None:
    priors = PredicatePriors.fit(
        torch.tensor([[1.0, 0.0]]),
        ("person",),
        ("ball",),
    )

    dense = priors.dense_logits(("person", "ball"))

    torch.testing.assert_close(dense[0, 1], priors.category_pair_logits[("person", "ball")])
    torch.testing.assert_close(dense[1, 0], priors.frequency_logits)


def test_oracle_complementarity_nests_the_category_prior() -> None:
    pair_logits = torch.tensor(
        [
            [[3.0, 1.0], [1.0, 3.0]],
            [[2.0, 1.0], [1.0, 2.0]],
        ]
    )
    model = PredicateComplementarity(2, pair_logits, condition="union-category-oracle")
    torch.nn.init.zeros_(model.union_readout.weight)
    torch.nn.init.zeros_(model.union_readout.bias)

    outputs = model(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        subject_categories=torch.tensor([0]),
        object_categories=torch.tensor([1]),
    )

    torch.testing.assert_close(outputs.predicate_logits, pair_logits[0, 1].log_softmax(0)[None])

    assert model.category_logit_scale is not None
    with torch.no_grad():
        model.category_logit_scale.zero_()
    union_only_outputs = model(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        subject_categories=torch.tensor([0]),
        object_categories=torch.tensor([1]),
    )
    torch.testing.assert_close(union_only_outputs.predicate_logits, torch.zeros(1, 2))


def test_predicted_categories_do_not_receive_predicate_gradients() -> None:
    model = PredicateComplementarity(
        2,
        torch.tensor(
            [
                [[3.0, 1.0], [1.0, 3.0]],
                [[2.0, 1.0], [1.0, 2.0]],
            ]
        ),
        condition="union-category-predicted",
    )
    outputs = model(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
    )

    F.cross_entropy(outputs.predicate_logits, torch.tensor([0])).backward()

    assert model.union_readout.weight.grad is not None
    assert model.category_readout is not None
    assert model.category_readout.weight.grad is None
