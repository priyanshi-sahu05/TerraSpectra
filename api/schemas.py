from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    class_id: int = Field(
        description="Predicted class ID"
    )

    class_name: str = Field(
        description="Predicted class name"
    )

    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of predicted class"
    )

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Chemical stress risk score"
    )