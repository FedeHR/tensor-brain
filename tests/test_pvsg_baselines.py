import torch

from experiments.pvsg.baselines import PredicatePriors


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
