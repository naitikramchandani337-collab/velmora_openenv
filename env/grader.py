def compute_reward(taken, expected, resources_left, max_resources):
    correct = 0

    for i in range(min(len(taken), len(expected))):
        if taken[i] == expected[i]:
            correct += 1
        else:
            break

    progress = correct / len(expected)
    efficiency = resources_left / max_resources

    penalty = 0.0
    if len(taken) > len(expected):
        penalty = 0.2

    score = 0.6 * progress + 0.3 * efficiency - penalty
    return max(0.0, min(1.0, score))