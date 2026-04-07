from velmora_env.models import Observation, Reward, State


class IncidentEnv:
    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self.current_index = 0
        self.incidents_completed = 0

        self.tasks = {
            "easy": [
                {
                    "incident": "Typo in homepage headline",
                    "hint": "Visible issue, no backend impact",
                    "logs": "Frontend content mismatch detected in homepage banner.",
                    "ideal_sequence": ["fix", "monitor"],
                    "requires_escalation": False,
                    "requires_containment": False,
                    "severity": "low"
                },
                {
                    "incident": "Broken image on landing page",
                    "hint": "Media asset failing to load",
                    "logs": "CDN asset 404 for marketing hero image.",
                    "ideal_sequence": ["investigate", "fix", "monitor"],
                    "requires_escalation": False,
                    "requires_containment": False,
                    "severity": "low"
                },
                {
                    "incident": "Forgotten footer link redirecting incorrectly",
                    "hint": "Minor navigation issue",
                    "logs": "Link target mismatch in footer navigation config.",
                    "ideal_sequence": ["investigate", "fix"],
                    "requires_escalation": False,
                    "requires_containment": False,
                    "severity": "low"
                }
            ],
            "medium": [
                {
                    "incident": "API latency spike affecting checkout",
                    "hint": "Users report slow checkout responses",
                    "logs": "p95 latency exceeds threshold for /checkout API.",
                    "ideal_sequence": ["investigate", "fix", "monitor"],
                    "requires_escalation": False,
                    "requires_containment": False,
                    "severity": "medium"
                },
                {
                    "incident": "Payment gateway intermittently failing",
                    "hint": "Transactions fail randomly",
                    "logs": "External gateway timeout and retry storm observed.",
                    "ideal_sequence": ["investigate", "escalate", "fix", "monitor"],
                    "requires_escalation": True,
                    "requires_containment": False,
                    "severity": "medium"
                },
                {
                    "incident": "Email notifications delayed",
                    "hint": "Queue backlog building up",
                    "logs": "Worker lag detected in notification queue consumers.",
                    "ideal_sequence": ["investigate", "fix", "monitor"],
                    "requires_escalation": False,
                    "requires_containment": False,
                    "severity": "medium"
                },
                {
                    "incident": "Customer dashboard occasionally crashes",
                    "hint": "Intermittent frontend instability",
                    "logs": "Frontend error burst correlated with stale API schema.",
                    "ideal_sequence": ["investigate", "fix", "monitor"],
                    "requires_escalation": False,
                    "requires_containment": False,
                    "severity": "medium"
                }
            ],
            "hard": [
                {
                    "incident": "Data breach suspected in user database",
                    "hint": "Unknown IPs accessing sensitive records",
                    "logs": "Security alert: abnormal access volume from foreign IP ranges.",
                    "ideal_sequence": ["contain", "investigate", "escalate", "monitor"],
                    "requires_escalation": True,
                    "requires_containment": True,
                    "severity": "high"
                },
                {
                    "incident": "Microservices failing after deployment",
                    "hint": "Multiple services returning 500 errors",
                    "logs": "Deployment rollback incomplete; service mesh errors cascading.",
                    "ideal_sequence": ["investigate", "contain", "fix", "monitor"],
                    "requires_escalation": False,
                    "requires_containment": True,
                    "severity": "high"
                },
                {
                    "incident": "Traffic spike causing instability",
                    "hint": "Possible DDoS or unexpected viral traffic",
                    "logs": "Inbound traffic exceeds autoscaling policy limits.",
                    "ideal_sequence": ["investigate", "contain", "monitor"],
                    "requires_escalation": True,
                    "requires_containment": True,
                    "severity": "high"
                },
                {
                    "incident": "Authentication service degraded across regions",
                    "hint": "Global login failures increasing rapidly",
                    "logs": "Auth replication lag and token validation failures observed.",
                    "ideal_sequence": ["investigate", "escalate", "contain", "fix", "monitor"],
                    "requires_escalation": True,
                    "requires_containment": True,
                    "severity": "high"
                }
            ]
        }

        self.state_data = {}

    def reset(self):
        self.current_index = 0
        self.incidents_completed = 0
        self.state_data = {
            "resources": 5,
            "steps_taken": [],
            "done": False
        }
        return self._get_observation()

    def _current_incident(self):
        if self.current_index >= len(self.tasks[self.task_name]):
            return None
        return self.tasks[self.task_name][self.current_index]

    def _infer_stage(self, taken):
        if not taken:
            return "initial"
        if "contain" in taken and "fix" not in taken:
            return "contained"
        if "investigate" in taken and "fix" not in taken:
            return "investigating"
        if "fix" in taken and "monitor" not in taken:
            return "mitigated"
        if "monitor" in taken:
            return "monitoring"
        return "active"

    def _get_observation(self):
        current = self._current_incident()

        if current is None:
            return Observation(
                incident="All incidents resolved",
                hint="Episode complete",
                logs="No further logs",
                user_impact="None",
                system_status="stable",
                steps_taken=self.state_data["steps_taken"],
                resources=self.state_data["resources"],
                severity="none",
                current_stage="complete"
            )

        severity = current["severity"]
        user_impact = "Low" if severity == "low" else "Moderate" if severity == "medium" else "Severe"
        system_status = "stable" if severity == "low" else "degraded" if severity == "medium" else "critical"

        return Observation(
            incident=current["incident"],
            hint=current["hint"],
            logs=current["logs"],
            user_impact=user_impact,
            system_status=system_status,
            steps_taken=self.state_data["steps_taken"],
            resources=self.state_data["resources"],
            severity=severity,
            current_stage=self._infer_stage(self.state_data["steps_taken"])
        )

    def step(self, action):
        if self.state_data["done"]:
            return self._get_observation(), Reward(score=0.0, feedback="Episode already complete", progress=1.0, penalty=0.0), True, {}

        current = self._current_incident()
        taken = self.state_data["steps_taken"]
        ideal = current["ideal_sequence"]

        repeated = action.action in taken
        taken.append(action.action)
        self.state_data["resources"] = max(self.state_data["resources"] - 1, 0)

        correct_action = action.action in ideal
        correct_prefix_len = 0
        for i, act in enumerate(taken):
            if i < len(ideal) and act == ideal[i]:
                correct_prefix_len += 1
            else:
                break

        progress = correct_prefix_len / len(ideal)
        efficiency = self.state_data["resources"] / 5.0

        penalty = 0.0

        if not correct_action:
            penalty += 0.20

        if repeated:
            penalty += 0.30

        if action.action == "investigate" and "investigate" in taken[:-1]:
            penalty += 0.20

        if action.action == "fix" and "investigate" not in taken[:-1]:
            penalty += 0.20

        if current["requires_containment"] and "contain" not in taken and current["severity"] == "high":
            penalty += 0.10

        if current["requires_escalation"] and action.action == "fix" and "escalate" not in taken and current["severity"] == "high":
            penalty += 0.15

        if action.action == "monitor" and "fix" not in taken and "contain" not in taken:
            penalty += 0.10

        if action.action == "fix" and "investigate" in taken[:-1] and current["severity"] == "low":
            penalty -= 0.05

        if self.state_data["resources"] == 0 and progress < 1.0:
            penalty += 0.20

        severity_weight = {
            "low": 0.9,
            "medium": 1.0,
            "high": 1.15
        }

        score = (
            0.55 * progress +
            0.25 * efficiency +
            0.20 * (1.0 if correct_action else 0.0)
        )

        score = score * severity_weight[current["severity"]]
        score -= penalty
        score = max(0.0, min(1.0, score))

        if correct_action:
            feedback = "Useful incident-response action."
        elif repeated:
            feedback = "Repeated action reduced efficiency."
        else:
            feedback = "Action was weak or out of sequence."

        done = False

        if progress >= 1.0 or self.state_data["resources"] <= 0:
            self.current_index += 1
            self.incidents_completed += 1 if progress >= 1.0 else 0

            if self.current_index >= len(self.tasks[self.task_name]):
                done = True
                self.state_data["done"] = True
                return (
                    self._get_observation(),
                    Reward(score=score, feedback=feedback, progress=progress, penalty=penalty),
                    done,
                    {
                        "final": True,
                        "progress": progress,
                        "incidents_completed": self.incidents_completed
                    }
                )

            self.state_data["resources"] = 5
            self.state_data["steps_taken"] = []

        return (
            self._get_observation(),
            Reward(score=score, feedback=feedback, progress=progress, penalty=penalty),
            done,
            {
                "progress": progress,
                "penalty": penalty,
                "incidents_completed": self.incidents_completed
            }
        )

    def state(self):
        return State(
            task_name=self.task_name,
            current_incident_index=self.current_index,
            resources=self.state_data["resources"],
            steps_taken=self.state_data["steps_taken"],
            done=self.state_data["done"],
            incidents_completed=self.incidents_completed
        )