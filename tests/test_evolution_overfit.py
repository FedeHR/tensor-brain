from experiments.evolution_overfit import train_variant


def test_all_evolution_variants_can_overfit_the_xor_diagnostic() -> None:
    for variant in ("original", "qtb-sigmoid", "qtb-relu"):
        result = train_variant(variant)

        assert result.accuracy == 1.0
        assert result.final_loss < 1e-3
