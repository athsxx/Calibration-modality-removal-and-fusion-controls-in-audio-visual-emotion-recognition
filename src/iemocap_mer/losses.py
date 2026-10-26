from __future__ import annotations

import torch
import torch.nn.functional as F


def concordance_correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean, target_mean = prediction.mean(0), target.mean(0)
    pred_var = prediction.var(0, unbiased=False)
    target_var = target.var(0, unbiased=False)
    covariance = ((prediction - pred_mean) * (target - target_mean)).mean(0)
    ccc = 2 * covariance / (
        pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8
    )
    return 1 - ccc.mean()


def orthogonality_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    shared = F.normalize(shared, dim=-1)
    private = F.normalize(private, dim=-1)
    return (shared * private).sum(-1).pow(2).mean()


def multitask_loss(
    output: dict,
    labels: torch.Tensor,
    va: torch.Tensor,
    weights: dict,
    has_va: bool = True,
) -> dict:
    losses = {
        "emotion": F.cross_entropy(output["logits"], labels),
    }
    if has_va:
        losses["va"] = concordance_correlation_loss(output["va"], va)
    if "audio_shared" in output:
        losses["alignment"] = 1 - F.cosine_similarity(
            output["audio_shared"], output["video_shared"], dim=-1
        ).mean()
        losses["orthogonality"] = 0.5 * (
            orthogonality_loss(output["audio_shared"], output["audio_private"])
            + orthogonality_loss(output["video_shared"], output["video_private"])
        )
        losses["audio_to_video_distillation"] = F.mse_loss(
            output["audio_teaches_video"], output["video_shared"].detach()
        )
        losses["video_to_audio_distillation"] = F.mse_loss(
            output["video_teaches_audio"], output["audio_shared"].detach()
        )
    total = sum(float(weights.get(name, 0.0)) * value for name, value in losses.items())
    return {"total": total, **losses}


# Ma et al., ICML 2023, Eq. (6): L = L_CL + λ L_CML, with Algorithm 1
# averaging the classification terms by 1/M. Appendix H.5 selects λ on a
# validation set from {1, 5, 10, ..., 100}. λ = 1.0 here is one fixed untuned
# setting. It was not chosen on these development or test sets, and it is not
# described as the authors' default.
CML_FIXED_WEIGHT = 1.0


def predictive_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability, the confidence in Ma et al. Eq. (4)."""
    return torch.softmax(logits, dim=-1).amax(dim=-1)


def cml_regularization_loss(conf_full: torch.Tensor, *removed: torch.Tensor) -> torch.Tensor:
    """Sum of batch-mean hinges mean(relu(conf_removed - conf_full)).

    One tensor per removed-modality variant, each compared with the full-input
    confidence. The sum is >= 0 and is 0 when every removed confidence is
    <= the full confidence (ties included; relu(0) = 0).
    """
    if not removed:
        raise ValueError("CML needs at least one removed-modality confidence")
    total = conf_full.new_zeros(())
    for conf_removed in removed:
        total = total + torch.relu(conf_removed - conf_full).mean()
    return total


def combine_task_and_cml(
    task_loss: torch.Tensor,
    cml_weight: float,
    penalty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``task_loss`` unchanged when the weight is 0."""
    weight = float(cml_weight)
    if weight == 0.0:
        return task_loss
    if weight < 0.0:
        raise ValueError("cml_weight must be >= 0")
    if penalty is None:
        raise ValueError("positive cml_weight requires a CML penalty")
    return task_loss + weight * penalty


def faithful_cml_objective(
    classification_full: torch.Tensor,
    classification_partials: list[torch.Tensor] | tuple[torch.Tensor, ...],
    conf_full: torch.Tensor,
    conf_partials: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    cml_weight: float,
) -> torch.Tensor:
    """Algorithm 1 objective for one sampled removal chain.

    For M modalities the chain has M - 1 partial inputs. The returned scalar is

        (1/M) * (CE(full) + CE(partial_1) + ... + CE(partial_{M-1}))
        + λ * sum_k mean(relu(Conf(T_k) - Conf(S_k)))

    Conf is the max softmax probability. Neither confidence is detached, so
    the hinge sends gradient into the full input and every partial input.
    The 1/M factor applies only to the summed classification losses. Each
    hinge is a batch mean, matching cross-entropy's mean reduction, and is
    not divided by M. Audio-visual training passes one partial input (M = 2).
    """
    n_partial = len(classification_partials)
    if n_partial < 1 or len(conf_partials) != n_partial:
        raise ValueError("Algorithm 1 needs one partial input per removed modality")
    weight = float(cml_weight)
    if weight < 0.0:
        raise ValueError("cml_weight must be >= 0")
    n_modalities = n_partial + 1
    total_classification = classification_full
    for partial in classification_partials:
        total_classification = total_classification + partial
    classification = total_classification / float(n_modalities)
    hinge = classification_full.new_zeros(())
    confidence_more = conf_full
    for confidence_fewer in conf_partials:
        hinge = hinge + torch.relu(confidence_fewer - confidence_more).mean()
        confidence_more = confidence_fewer
    return classification + weight * hinge


def faithful_training_loss(
    task_total: torch.Tensor,
    emotion_unweighted: torch.Tensor,
    emotion_weight: float,
    faithful_term: torch.Tensor,
) -> torch.Tensor:
    """Replace the full-input classification term with the Algorithm 1 term.

    Non-classification terms already inside ``task_total`` (valence-activation
    concordance on IEMOCAP) stay on the full input and are not divided by M.
    """
    return task_total - float(emotion_weight) * emotion_unweighted + faithful_term


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, emotion_weight: float) -> torch.Tensor:
    return float(emotion_weight) * F.cross_entropy(logits, labels)
