from env.environment import IncidentEnv
from env.models import Action

game = IncidentEnv(task_name="easy")

print("Creating env done")

result = game.reset()
print("RESET OUTPUT:", result)

step_result = game.step(Action(action="hello"))
print("STEP OUTPUT:", step_result)