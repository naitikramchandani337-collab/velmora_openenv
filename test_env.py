from velmora_env.environment import IncidentEnv
from velmora_env.models import Action

for level in ["easy", "medium", "hard"]:
    print(f"\n=== LEVEL: {level} ===")
    game = IncidentEnv(task_name=level)
    print("Creating env done")
    result = game.reset()
    print("RESET OUTPUT:", result)
    step_result = game.step(Action(action="investigate"))
    print("STEP OUTPUT:", step_result)
