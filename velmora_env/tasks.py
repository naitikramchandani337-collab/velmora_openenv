def get_task(level):

    if level == "easy":
        return {
            "resources": 3,
            "subtasks": [
                {
                    "description": "Website page not loading for some users",
                    "hint": "This seems like a minor issue affecting a small group.",
                    "solution": ["investigate", "fix"]
                },
                {
                    "description": "Typo found on homepage content",
                    "hint": "Issue is clearly visible and isolated.",
                    "solution": ["fix"]
                }
            ]
        }

    elif level == "medium":
        return {
            "resources": 3,
            "subtasks": [
                {
                    "description": "API responding slowly",
                    "hint": "Root cause unclear, needs diagnosis.",
                    "solution": ["investigate", "fix"]
                },
                {
                    "description": "Memory usage increasing continuously",
                    "hint": "May affect system stability soon.",
                    "solution": ["investigate", "contain"]
                },
                {
                    "description": "Users reporting intermittent failures",
                    "hint": "System still running but unstable.",
                    "solution": ["investigate", "monitor"]
                }
            ]
        }

    elif level == "hard":
        return {
            "resources": 2,
            "subtasks": [
                {
                    "description": "Multiple services are down",
                    "hint": "High severity incident.",
                    "solution": ["contain", "investigate", "fix"]
                },
                {
                    "description": "Possible security breach detected",
                    "hint": "Immediate escalation required.",
                    "solution": ["contain", "escalate"]
                },
                {
                    "description": "System unstable after fix",
                    "hint": "Underlying issue still exists.",
                    "solution": ["investigate", "monitor"]
                },
                {
                    "description": "Repeated customer complaints",
                    "hint": "Business impact high.",
                    "solution": ["escalate"]
                }
            ]
        }