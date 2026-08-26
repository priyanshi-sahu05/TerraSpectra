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

class PredictionRequest(BaseModel):

    data: list[float] = Field(
        description="Flattened hyperspectral input data"
    )

    channels: int = Field(
        gt=0,
        description="Number of input channels"
    )

    depth: int = Field(
        gt=0,
        description="Depth of the input tensor"
    )

    height: int = Field(
        gt=0,
        description="Height of the input tensor"
    )

    width: int = Field(
        gt=0,
        description="Width of the input tensor"
    )

class TilePrediction(BaseModel):

    tile_id: int

    row: int

    col: int

    height: int

    width: int

    class_id: int

    class_name: str

    probability: float = Field(
        ge=0.0,
        le=1.0,
    )


class TiledPredictionResponse(BaseModel):

    total_tiles: int

    healthy_tiles: int

    stressed_tiles: int

    stressed_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    average_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    overall_class_id: int

    overall_class_name: str

    tiles: list[TilePrediction]