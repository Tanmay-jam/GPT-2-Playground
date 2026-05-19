import torch
from dataclasses import dataclass


@dataclass
class DraftStep:
    token_id: int
    token_text: str
    draft_prob: float
    target_prob: float
    acceptance_prob: float
    accepted: bool
    resampled_token_id: int = None
    resampled_token_text: str = None


def run_speculative_decoding(target_model, draft_model, input_ids, tokenizer, K=5, temperature=1.0):
    """
    One round of speculative decoding.

    Draft model greedily generates K tokens. Target model verifies all K in
    one forward pass. Tokens are accepted with probability min(1, p_target / p_draft).
    First rejection triggers a resample from the adjusted distribution and stops.
    If all K accepted, one bonus token is sampled from the target.

    Returns:
        steps: list of DraftStep (one per draft token)
        accepted_ids: final list of accepted token ids (length <= K+1)
    """
    # ── Step 1: Draft generates K tokens greedily ──────────────────────────
    draft_ids = []
    draft_dists = []

    current_ids = input_ids.clone()
    for _ in range(K):
        with torch.no_grad():
            logits = draft_model.get_logits(current_ids)[0, -1]
        if temperature != 1.0:
            logits = logits / temperature
        prob_dist = torch.softmax(logits, dim=-1)
        token_id = torch.argmax(prob_dist).item()
        draft_ids.append(token_id)
        draft_dists.append(prob_dist)
        current_ids = torch.cat([current_ids, torch.tensor([[token_id]])], dim=1)

    # ── Step 2: Target verifies all K tokens in ONE forward pass ───────────
    full_ids = torch.cat([input_ids, torch.tensor([draft_ids])], dim=1)
    with torch.no_grad():
        target_logits_all = target_model.get_logits(full_ids)

    # ── Step 3: Accept / reject ────────────────────────────────────────────
    n_input = input_ids.shape[1]
    accepted_ids = []
    steps = []

    for k in range(K):
        # Position (n_input - 1 + k) in target output predicts draft_ids[k]
        target_logit = target_logits_all[0, n_input - 1 + k]
        if temperature != 1.0:
            target_logit = target_logit / temperature
        target_dist = torch.softmax(target_logit, dim=-1)

        token_id = draft_ids[k]
        draft_p = draft_dists[k][token_id].item()
        target_p = target_dist[token_id].item()
        acceptance_prob = min(1.0, target_p / (draft_p + 1e-10))
        accepted = torch.rand(1).item() < acceptance_prob

        step = DraftStep(
            token_id=token_id,
            token_text=tokenizer.decode([token_id]),
            draft_prob=draft_p,
            target_prob=target_p,
            acceptance_prob=acceptance_prob,
            accepted=accepted,
        )

        if accepted:
            accepted_ids.append(token_id)
            steps.append(step)
        else:
            # Resample from adjusted distribution: (target - draft)+
            adjusted = torch.clamp(target_dist - draft_dists[k], min=0.0)
            s = adjusted.sum()
            adjusted = adjusted / s if s > 0 else target_dist
            resampled_id = torch.multinomial(adjusted, num_samples=1).item()
            step.resampled_token_id = resampled_id
            step.resampled_token_text = tokenizer.decode([resampled_id])
            accepted_ids.append(resampled_id)
            steps.append(step)
            break
    else:
        # All K accepted — bonus token from target
        bonus_logit = target_logits_all[0, n_input - 1 + K]
        if temperature != 1.0:
            bonus_logit = bonus_logit / temperature
        bonus_probs = torch.softmax(bonus_logit, dim=-1)
        bonus_id = torch.multinomial(bonus_probs, num_samples=1).item()
        accepted_ids.append(bonus_id)

    return steps, accepted_ids
