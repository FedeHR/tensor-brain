"""Two learning stages for the Tensor Brain agent.

Stage A, behavioural cloning, is a *capacity* diagnostic in the spirit of this
repository's existing overfit gates: it asks whether the architecture can carry
a cue-conditioned policy at all, without reinforcement-learning variance.

Stage B, REINFORCE, is the agentic experiment proper. No new model component is
introduced for it: the policy is the generative measurement over the action
candidate group, so ``log pi(a|q)`` comes straight out of ``TensorBrain.measure``
and the value baseline out of ``TensorBrain.index_scores``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor

from experiments.agency.agent import GridAgent
from experiments.agency.gridworld import SymbolicForaging
from experiments.agency.rollout import EpisodeMetrics, evaluate, run_episodes, summarize


@dataclass(frozen=True)
class CloneConfig:
    updates: int = 400
    learning_rate: float = 3e-4
    percept_weight: float = 1.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class ReinforceConfig:
    updates: int = 1500
    learning_rate: float = 3e-4
    discount: float = 0.99
    entropy_weight: float = 0.02
    value_weight: float = 0.5
    percept_weight: float = 0.0
    percept_in_policy_gradient: bool = False
    normalize_advantage: bool = True
    grad_clip: float = 1.0
    evaluate_every: int = 25
    evaluate_repeats: int = 2


@dataclass
class TrainingLog:
    """Time series recorded during one training run."""

    update: list[int] = field(default_factory=list)
    episodes: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    train_metrics: list[dict[str, float]] = field(default_factory=list)
    eval_metrics: list[dict[str, float]] = field(default_factory=list)
    holdout_metrics: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def clone_from_oracle(
    environment: SymbolicForaging,
    agent: GridAgent,
    config: CloneConfig,
    *,
    evaluation: Callable[[], dict[str, dict[str, float]]] | None = None,
    evaluate_every: int = 25,
) -> TrainingLog:
    """Stage A: teacher-forced cloning of the privileged greedy oracle.

    The oracle knows where the cued object is; the agent does not. Cloning
    therefore does not produce an optimal POMDP policy, but a high cloning
    accuracy shows that the cue indices and the egocentric view *can* drive the
    action measurement, which is the capacity question.
    """

    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    log = TrainingLog()
    for update in range(config.updates):
        optimizer.zero_grad()
        batch = run_episodes(
            environment,
            agent,
            teacher_forcing=True,
            supervise_percepts=agent.config.measure_percepts,
        )
        alive = batch.alive.float()
        action_loss = -_masked_mean(batch.action_log_probability, alive)
        percept_loss = -_masked_mean(batch.percept_log_probability, alive)
        loss = action_loss + config.percept_weight * percept_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)
        optimizer.step()

        if update % evaluate_every == 0 or update == config.updates - 1:
            log.update.append(update)
            log.episodes.append((update + 1) * environment.num_envs)
            log.loss.append(float(loss.detach()))
            log.train_metrics.append(summarize(batch).as_dict())
            splits = evaluation() if evaluation is not None else {}
            log.eval_metrics.append(splits.get("eval", {}))
            log.holdout_metrics.append(splits.get("holdout", {}))
    return log


def reinforce(
    environment: SymbolicForaging,
    agent: GridAgent,
    config: ReinforceConfig,
    *,
    evaluation: Callable[[], dict[str, dict[str, float]]] | None = None,
) -> TrainingLog:
    r"""Stage B: policy gradient on the action measurement.

    .. math::
        L = -\overline{\log p_a (R - v)} - c_H \overline{H(p_a)}
            + c_V \overline{(v - R)^2} - c_P \overline{\log p_{percept}}

    ``v`` is the internal reward function. With the default
    ``critic="reward-index"`` it is the score of the ``reward_positive`` index,
    so the critic adds one column to ``A`` and nothing else. The advantage is
    detached from the critic, as usual, so the value term does not push the
    policy.
    """

    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    log = TrainingLog()
    for update in range(config.updates):
        optimizer.zero_grad()
        batch = run_episodes(environment, agent)
        alive = batch.alive.float()
        returns = batch.returns_to_go(config.discount)
        advantage = (returns - batch.value).detach()
        if config.normalize_advantage:
            # Standard variance reduction. It rescales the gradient only; it is
            # an optimization detail, not a Tensor Brain claim.
            live = alive > 0
            centred = advantage[live]
            advantage = (advantage - centred.mean()) / centred.std().clamp_min(1e-6)

        policy_loss = -_masked_mean(batch.action_log_probability * advantage, alive)
        if config.percept_in_policy_gradient:
            policy_loss = policy_loss - _masked_mean(
                batch.percept_log_probability * advantage, alive
            )
        entropy_loss = -_masked_mean(batch.action_entropy, alive)
        value_loss = _masked_mean((batch.value - returns) ** 2, alive)
        percept_loss = -_masked_mean(batch.percept_log_probability, alive)
        loss = (
            policy_loss
            + config.entropy_weight * entropy_loss
            + config.value_weight * value_loss
            + config.percept_weight * percept_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)
        optimizer.step()

        if update % config.evaluate_every == 0 or update == config.updates - 1:
            log.update.append(update)
            log.episodes.append((update + 1) * environment.num_envs)
            log.loss.append(float(loss.detach()))
            log.train_metrics.append(summarize(batch).as_dict())
            splits = evaluation() if evaluation is not None else {}
            log.eval_metrics.append(splits.get("eval", {}))
            log.holdout_metrics.append(splits.get("holdout", {}))
    return log


def split_evaluator(
    agent: GridAgent,
    train_environment: SymbolicForaging,
    holdout_environment: SymbolicForaging,
    repeats: int = 2,
) -> Callable[[], dict[str, dict[str, float]]]:
    """Evaluate on training cues and on the held-out cue diagonal."""

    def run() -> dict[str, dict[str, float]]:
        return {
            "eval": evaluate(train_environment, agent, repeats=repeats).as_dict(),
            "holdout": evaluate(holdout_environment, agent, repeats=repeats).as_dict(),
        }

    return run


def final_metrics(
    agent: GridAgent,
    train_environment: SymbolicForaging,
    holdout_environment: SymbolicForaging,
    repeats: int = 8,
) -> dict[str, EpisodeMetrics]:
    """Larger-sample evaluation used for the reported ablation table."""

    return {
        "eval": evaluate(train_environment, agent, repeats=repeats),
        "holdout": evaluate(holdout_environment, agent, repeats=repeats),
    }


__all__ = [
    "CloneConfig",
    "ReinforceConfig",
    "TrainingLog",
    "clone_from_oracle",
    "final_metrics",
    "reinforce",
    "split_evaluator",
]
