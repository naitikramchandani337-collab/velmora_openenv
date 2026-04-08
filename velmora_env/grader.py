def grade_task(env, task_name: str) -> float:
    total_incidents = len(env.tasks[task_name])
    incidents_completed = env.incidents_completed

    if total_incidents == 0:
        return 0.0

    completion_ratio = incidents_completed / total_incidents

    resources_left = env.state_data.get("resources", 0)
    max_resources = 5
    efficiency = resources_left / max_resources

    steps_taken_total = len(env.state_data.get("steps_taken", []))
    step_efficiency = max(0.0, 1.0 - (steps_taken_total / (total_incidents * max_resources)))

    difficulty_weight = {
        "easy": 1.0,
        "medium": 1.05,
        "hard": 1.10
    }

    raw_score = (
        0.50 * completion_ratio +
        0.25 * efficiency +
        0.25 * step_efficiency
    )

    score = raw_score * difficulty_weight.get(task_name, 1.0)
    return max(0.0, min(1.0, score))