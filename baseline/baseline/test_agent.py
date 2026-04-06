import os
from openai import OpenAI
from env.environment import IncidentEnv
from env.models import Action

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_agent(task="easy"):
    env = IncidentEnv(task_name=task)
    obs = env.reset()

    done = False
    total_score = 0

    while not done:
        prompt = f"""
        You are an incident response agent.

        Incident: {obs.incident}
        Hint: {obs.hint}
        Logs: {obs.logs}

        Available actions:
        investigate, fix, monitor, escalate, contain

        Choose the BEST next action.
        Only return one word.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        action_text = response.choices[0].message.content.strip().lower()

        action = Action(action=action_text)

        obs, reward, done, _ = env.step(action)

        total_score += reward.score

        print(f"Action: {action_text} | Score: {reward.score}")

    print("FINAL SCORE:", total_score)


if __name__ == "__main__":
    run_agent("easy")