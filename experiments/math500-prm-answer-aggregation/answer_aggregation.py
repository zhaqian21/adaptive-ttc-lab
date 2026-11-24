"""Answer grouping and PRM selection shared by live inference and replay."""

import math


def answer_key(answer):
    if isinstance(answer, str):
        return answer.strip() or None
    return None


def group_answers(predictions, rewards):
    """Group by extracted answer and keep the strongest representative per group."""
    members = {}
    for index, prediction in enumerate(predictions):
        answer = answer_key(prediction)
        if answer is not None:
            members.setdefault(answer, []).append(index)
    groups = []
    for answer, indices in members.items():
        best_index = max(indices, key=rewards.__getitem__)
        groups.append({
            "answer": answer,
            "members": indices,
            "support": len(indices),
            "sum_prm": math.fsum(rewards[index] for index in indices),
            "best_reward": rewards[best_index],
            "representative": best_index,
        })
    return groups


def winner(groups, field, original):
    if not groups or (field == "sum_prm" and not any(group[field] for group in groups)):
        return original
    group = max(groups, key=lambda g: (g[field], g["best_reward"], -g["representative"]))
    return group["representative"]
