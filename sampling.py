import torch


def sample_next_token(logits, mode, top_k, top_p, min_p, temperature, use_temperature) -> int:
    if use_temperature:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    if mode == "top-k":
        topk = torch.topk(probs, k=top_k)
        filtered = torch.zeros_like(probs)
        filtered[topk.indices] = topk.values

    elif mode == "top-p":
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative_probs - sorted_probs > top_p] = 0.0
        filtered = torch.zeros_like(probs)
        filtered[sorted_indices] = sorted_probs

    elif mode == "min-p":
        threshold = min_p * probs.max()
        filtered = probs.clone()
        filtered[filtered < threshold] = 0.0

    filtered /= filtered.sum()
    return torch.multinomial(filtered, num_samples=1).item()


def get_filtered_probs(logits, mode, top_k, top_p, min_p, temperature) -> tuple:
    """Returns (token_indices, probabilities) after filtering, for visualization."""
    probs = torch.softmax(logits / temperature if temperature != 1.0 else logits, dim=-1)

    if mode == "top-k":
        topk = torch.topk(probs, k=top_k)
        values = topk.values / topk.values.sum()
        return topk.indices, values

    elif mode == "top-p":
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs <= top_p
        indices = sorted_indices[mask]
        values = sorted_probs[mask]
        return indices, values / values.sum()

    elif mode == "min-p":
        threshold = min_p * probs.max()
        mask = probs >= threshold
        indices = mask.nonzero(as_tuple=True)[0]
        values = probs[indices]
        sorted_order = torch.argsort(values, descending=True)
        indices = indices[sorted_order]
        values = values[sorted_order]
        return indices, values / values.sum()
