from pydantic import BaseModel, Field
from typing import List


# -------------------------
# ACTION
# -------------------------
class Action(BaseModel):
    action: str = Field(
        ...,
        description="Action taken by the agent",
        example="investigate"
    )


# -------------------------
# OBSERVATION
# -------------------------
class Observation(BaseModel):
    incident: str = Field(
        ...,
        description="Description of the current incident"
    )

    hint: str = Field(
        ...,
        description="Hint to guide the agent"
    )

    resources: int = Field(
        ...,
        description="Remaining actions/resources available"
    )

    steps_taken: List[str] = Field(
        ...,
        description="List of actions already taken"
    )

    # ✅ IMPORTANT (fixes your error + improves environment)
    logs: str = Field(
        ...,
        description="Simulated system logs related to the incident"
    )

    user_impact: str = Field(
        ...,
        description="Impact level on users (Low, Moderate, Severe)"
    )

    system_status: str = Field(
        ...,
        description="Current system health status (stable, degraded, critical)"
    )


# -------------------------
# REWARD
# -------------------------
class Reward(BaseModel):
    score: float = Field(
        ...,
        description="Reward score between 0.0 and 1.0"
    )

    feedback: str = Field(
        ...,
        description="Feedback explaining the reward"
    )