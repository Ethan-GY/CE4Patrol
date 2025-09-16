def action_score(predicted_actions, ground_actions):
    set_pred = set(predicted_actions)
    set_true = set(ground_actions)
    intersection = len(set_pred & set_true)
    union = len(set_pred | set_true)
    return intersection / union if union > 0 else 0.0