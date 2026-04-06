from env.models import Observation, Reward


class IncidentEnv:
    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self.current_index = 0

        self.tasks = {
            "easy": [
                {
                    "incident": "Typo in homepage headline",
                    "hint": "Visible issue, no system impact",
                    "correct_actions": ["fix"],
                    "severity": "low"
                },
                {
                    "incident": "Broken image on landing page",
                    "hint": "Media not loading properly",
                    "correct_actions": ["investigate", "fix"],
                    "severity": "low"
                }
            ],
            "medium": [
                {
                    "incident": "API latency spike affecting checkout",
                    "hint": "Users reporting slow checkout",
                    "correct_actions": ["investigate", "fix", "monitor"],
                    "severity": "medium"
                },
                {
                    "incident": "Payment gateway intermittently failing",
                    "hint": "Transactions failing randomly",
                    "correct_actions": ["investigate", "escalate", "fix"],
                    "severity": "medium"
                },
                {
                    "incident": "Email notifications delayed",
                    "hint": "Queue processing is slow",
                    "correct_actions": ["investigate", "fix", "monitor"],
                    "severity": "medium"
                }
            ],
            "hard": [
                {
                    "incident": "Data breach suspected in user database",
                    "hint": "Unusual access logs detected from unknown IPs",
                    "correct_actions": ["contain", "investigate", "escalate", "monitor"],
                    "severity": "high"
                },
                {
                    "incident": "Microservices failing after deployment",
                    "hint": "Multiple services returning 500 errors",
                    "correct_actions": ["investigate", "contain", "fix", "monitor"],
                    "severity": "high"
                },
                {
                    "incident": "Traffic spike causing instability",
                    "hint": "Possible DDoS or viral traffic",
                    "correct_actions": ["investigate", "contain", "monitor"],
                    "severity": "high"
                }
            ]
        }

        self.state_data = {}

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.current_index = 0
        self.state_data = {
            "resources": 3,
            "steps_taken": [],
            "done": False
        }
        return self._get_observation()

    # -------------------------
    # OBSERVATION
    # -------------------------
    def _get_observation(self):
        if self.current_index >= len(self.tasks[self.task_name]):
            return Observation(
                incident="All incidents resolved",
                hint="Episode complete",
                resources=self.state_data["resources"],
                steps_taken=self.state_data["steps_taken"],
                logs="No further logs",
                user_impact="None",
                system_status="stable",
            )

        current = self.tasks[self.task_name][self.current_index]

        return Observation(
            incident=current["incident"],
            hint=current["hint"],
            resources=self.state_data["resources"],
            steps_taken=self.state_data["steps_taken"],
            logs=f"System logs indicate: {current['hint']}",
            user_impact=(
                "Low"
                if current["severity"] == "low"
                else "Moderate"
                if current["severity"] == "medium"
                else "Severe"
            ),
            system_status=(
                "stable"
                if current["severity"] == "low"
                else "degraded"
                if current["severity"] == "medium"
                else "critical"
            ),
        )

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):
        if self.state_data["done"]:
            return self._get_observation(), Reward(score=0.0, feedback="Episode done"), True, {}

        current = self.tasks[self.task_name][self.current_index]

        correct = action.action in current["correct_actions"]
        repeated = action.action in self.state_data["steps_taken"]

        self.state_data["steps_taken"].append(action.action)
        self.state_data["resources"] -= 1
        self.state_data["resources"] = max(self.state_data["resources"], 0)

        # -------------------------
        # PROGRESS & EFFICIENCY
        # -------------------------
        progress = len(
            set(self.state_data["steps_taken"]) &
            set(current["correct_actions"])
        ) / len(current["correct_actions"])

        efficiency = self.state_data["resources"] / 3

        # -------------------------
        # PENALTIES (IMPROVED 🔥)
        # -------------------------
        penalty = 0

        # ❌ Wrong action penalty
        if not correct:
            penalty += 0.25

        # 🔁 Repeated action penalty
        if repeated:
            penalty += 0.15

        # ⚠️ Bad sequence penalty (important for judges)
        if current["severity"] == "high":
            if action.action == "escalate" and "investigate" not in self.state_data["steps_taken"]:
                penalty += 0.35

        # 🧠 Resource misuse penalty
        if self.state_data["resources"] == 0 and progress < 1.0:
            penalty += 0.2

        severity_weight = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.2,
        }

        time_penalty = 0.05 * (3 - self.state_data["resources"])

        # -------------------------
        # SCORE
        # -------------------------
        score = (
            0.5 * progress +
            0.3 * efficiency +
            0.2 * (1 if correct else 0)
        )

        score = score * severity_weight[current["severity"]] - time_penalty
        score -= penalty
        score = max(0.0, min(1.0, score))

        # -------------------------
        # FEEDBACK
        # -------------------------
        if correct:
            feedback = "Effective step toward resolution."
        elif repeated:
            feedback = "Repeated action detected."
        else:
            feedback = "Incorrect action for this incident."

        done = False

        # -------------------------
        # TASK TRANSITION
        # -------------------------
        if progress >= 1.0 or self.state_data["resources"] <= 0:
            self.current_index += 1

            if self.current_index >= len(self.tasks[self.task_name]):
                done = True
                self.state_data["done"] = True

                return (
                    self._get_observation(),
                    Reward(score=score, feedback=feedback),
                    done,
                    {"final": True, "progress": progress}
                )

            self.state_data["resources"] = 3
            self.state_data["steps_taken"] = []

        return (
            self._get_observation(),
            Reward(score=score, feedback=feedback),
            done,
            {"progress": progress, "penalty": penalty}
        )

    # -------------------------
    # STATE
    # -------------------------
    def state(self):
        return self.state_data