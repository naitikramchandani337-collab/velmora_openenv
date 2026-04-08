import os
from velmora_env.environment import IncidentEnv

CLASSIFIER = os.getenv("BASELINE_CLASSIFIER", "groq").lower()  # "groq" or "xgb"

VALID_ACTIONS = ["investigate", "fix", "monitor", "escalate", "contain"]


def choose_action_groq(observation):
    from openai import OpenAI
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY", os.getenv("HF_TOKEN")),
    )
    model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    prompt = (
        f"You are an expert incident response agent. Your job is to pick the single best next action.\n\n"
        f"Available actions: {VALID_ACTIONS}\n"
        f"Rules:\n"
        f"- For LOW severity: if the cause is obvious (typo, broken link, visual issue), go straight to 'fix' then 'monitor'. If the cause is unclear (asset loading, CDN, config), start with 'investigate' first.\n"
        f"- For MEDIUM severity: always start with 'investigate'. Only use 'escalate' if logs mention external/third-party/vendor/gateway issues. Then 'fix', then 'monitor'.\n"
        f"- For HIGH severity: start with 'contain' for security/breach/data issues, then 'investigate', then 'escalate', then 'fix', then 'monitor'\n"
        f"- Never repeat an action already taken. Never add extra actions beyond what is needed.\n\n"
        f"Incident: {observation.incident}\n"
        f"Severity: {observation.severity}\n"
        f"Hint: {observation.hint}\n"
        f"Logs: {observation.logs}\n"
        f"Steps already taken: {observation.steps_taken}\n\n"
        f"Respond with exactly one word from the available actions list. No explanation."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )

    action = response.choices[0].message.content.strip().lower()
    return action if action in VALID_ACTIONS else "investigate"


def choose_action_xgb(observation):
    import numpy as np
    from xgboost import XGBClassifier
    import joblib

    model_path = os.getenv("XGB_MODEL_PATH", "baseline/xgb_model.pkl")
    clf = joblib.load(model_path)

    keywords = VALID_ACTIONS + ["breach", "payment", "latency", "typo", "image", "access", "slow", "media"]
    text = f"{observation.incident} {observation.hint} {observation.severity}".lower()
    features = np.array([[1 if kw in text else 0 for kw in keywords]], dtype=np.float32)

    label = clf.predict(features)[0]
    return VALID_ACTIONS[int(label)] if int(label) < len(VALID_ACTIONS) else "investigate"


def choose_action(observation):
    if CLASSIFIER == "xgb":
        return choose_action_xgb(observation)
    return choose_action_groq(observation)


def run_task(task_name, debug=False):
    env = IncidentEnv(task_name=task_name)
    obs = env.reset()

    done = False
    total_score = 0
    steps = 0

    while not done:
        action_text = choose_action(obs)
        if debug:
            print(f"  incident={obs.incident!r} severity={obs.severity} taken={obs.steps_taken} -> {action_text}")
        action = type("Action", (), {"action": action_text})()
        obs, reward, done, _ = env.step(action)
        total_score += reward.score
        steps += 1

    final_score = total_score / steps if steps > 0 else 0.0
    return round(final_score, 3)


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        score = run_task(task, debug=True)
        print(task, score)
