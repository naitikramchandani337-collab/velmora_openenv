import os
from openai import OpenAI
from env.environment import IncidentEnv
from env.models import Action

# ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)


# -------------------------
# LLM AGENT
# -------------------------
def get_llm_action(observation):
    prompt = f"""
You are an AI incident response agent.

Your goal is to resolve the incident efficiently using the correct sequence of actions.

Available actions:
- investigate
- fix
- monitor
- escalate
- contain

Rules:
- Always investigate before fixing complex issues
- For high severity issues, contain first, then investigate, then escalate if needed
- Avoid repeating actions
- Use resources efficiently

Incident:
{observation.incident}

Hint:
{observation.hint}

Logs:
{observation.logs}

Steps taken:
{observation.steps_taken}

Resources left:
{observation.resources}

Return ONLY one action from:
investigate, fix, monitor, escalate, contain
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip().lower()


# -------------------------
# RUN TASK
# -------------------------
def run_task(task_name):
    env = IncidentEnv(task_name)
    obs = env.reset()

    done = False
    total_score = 0
    step_count = 0

    print(f"[START] task={task_name}")

    while not done:
        action_text = get_llm_action(obs)

        # safety fallback
        if action_text not in ["investigate", "fix", "monitor", "escalate", "contain"]:
            action_text = "investigate"

        action = Action(action=action_text)

        obs, reward, done, info = env.step(action)

        total_score += reward.score
        step_count += 1

        print(f"[STEP] task={task_name} step={step_count} action={action_text} score={reward.score:.3f}")

    final_score = total_score / step_count if step_count > 0 else 0.0

    print(f"[END] task={task_name} score={final_score:.3f}")

    return final_score


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    run_task("easy")
    run_task("medium")
    run_task("hard")