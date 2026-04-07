def grade_task(env, task_name: str) -> float:
    total_incidents = len(env.tasks[task_name])
    incidents_completed = env.incidents_completed

    completion_ratio = incidents_completed / total_incidents if total_incidents > 0 else 0.0

    resources_left = env.state_data.get("resources", 0)
    efficiency = resources_left / 5.0

    difficulty_bonus = {
        "easy": 0.0,
        "medium": 0.05,
        "hard": 0.10
    }

    score = 0.8 * completion_ratio + 0.2 * efficiency + difficulty_bonus.get(task_name, 0.0)
    return max(0.0, min(1.0, score))