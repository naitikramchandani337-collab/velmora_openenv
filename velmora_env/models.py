from pydantic import BaseModel, Field
from typing import List


class Action(BaseModel):
    action: str = Field(
        ...,
        description="Action taken by the agent",
        examples=["investigate"]
    )


class Observation(BaseModel):
    incident: str = Field(..., description="Description of the current incident")
    hint: str = Field(..., description="Hint to guide the agent")
    logs: str = Field(..., description="Simulated system logs related to the incident")
    user_impact: str = Field(..., description="Impact level on users")
    system_status: str = Field(..., description="Current system health status")
    steps_taken: List[str] = Field(..., description="Actions already taken")
    resources: int = Field(..., description="Remaining action budget")
    severity: str = Field(..., description="Incident severity level")
    current_stage: str = Field(..., description="Current inferred stage of incident handling")


class Reward(BaseModel):
    score: float = Field(..., description="Reward score between 0.0 and 1.0")
    feedback: str = Field(..., description="Explanation of reward")
    progress: float = Field(..., description="Normalized progress toward resolution")
    penalty: float = Field(..., description="Penalty applied this step")


class State(BaseModel):
    task_name: str
    current_incident_index: int
    resources: int
    steps_taken: List[str]
    done: bool
    incidents_completed: int