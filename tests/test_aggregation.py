from inference.aggregation import (
    aggregate_tile_predictions,
)


def test_aggregate_tile_predictions():

    predictions = [
        {
            "tile_id": 0,
            "row": 0,
            "col": 0,
            "height": 8,
            "width": 8,
            "class_id": 1,
            "probability": 0.80,
        },
        {
            "tile_id": 1,
            "row": 0,
            "col": 8,
            "height": 8,
            "width": 8,
            "class_id": 1,
            "probability": 0.70,
        },
        {
            "tile_id": 2,
            "row": 8,
            "col": 0,
            "height": 8,
            "width": 8,
            "class_id": 0,
            "probability": 0.60,
        },
        {
            "tile_id": 3,
            "row": 8,
            "col": 8,
            "height": 8,
            "width": 8,
            "class_id": 0,
            "probability": 0.50,
        },
    ]

    result = aggregate_tile_predictions(
        predictions
    )

    assert result["total_tiles"] == 4

    assert result["healthy_tiles"] == 2

    assert result["stressed_tiles"] == 2

    assert result["stressed_ratio"] == 0.5

    assert result["overall_class_id"] == 1

    assert (
        result["overall_class_name"]
        == "Chemically Stressed"
    )

    assert (
        result["average_probability"]
        == 0.65
    )


def test_empty_predictions():

    try:

        aggregate_tile_predictions([])

        assert False

    except ValueError:

        assert True