from velmora_env.environment import IncidentEnv


def choose_action(observation):
    """
    Simple rule-based agent:
    Uses hint keywords to decide next action
    """

    hint = observation.hint.lower()

    if "typo" in hint or "visible" in hint:
        return "fix"

    if "image" in hint or "media" in hint:
        if "investigate" not in observation.steps_taken:
            return "investigate"
        return "fix"

    if "latency" in hint or "slow" in hint:
        if "investigate" not in observation.steps_taken:
            return "investigate"
        if "fix" not in observation.steps_taken:
            return "fix"
        return "monitor"

    if "payment" in hint:
        if "investigate" not in observation.steps_taken:
            return "investigate"
        if "escalate" not in observation.steps_taken:
            return "escalate"
        return "fix"

    if "breach" in hint or "access" in hint:
        if "contain" not in observation.steps_taken:
            return "contain"
        if "investigate" not in observation.steps_taken:
            return "investigate"
        if "escalate" not in observation.steps_taken:
            return "escalate"
        return "monitor"

    return "investigate"


def run_task(task_name):
    env = IncidentEnv(task_name=task_name)
    obs = env.reset()

    done = False
    total_score = 0
    steps = 0

    while not done:
        action_text = choose_action(obs)

        action = type("Action", (), {"action": action_text})()

        obs, reward, done, _ = env.step(action)

        total_score += reward.score
        steps += 1

    # ✅ NORMALIZED FINAL SCORE (IMPORTANT FIX)
    final_score = total_score / steps if steps > 0 else 0.0

    return round(final_score, 3)


if __name__ == "__main__":
    print("easy", run_task("easy"))
    print("medium", run_task("medium"))
    print("hard", run_task("hard"))