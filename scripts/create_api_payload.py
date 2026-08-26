import json
import random


def main():

    channels = 1
    depth = 8
    height = 8
    width = 8

    total_values = (
        channels
        * depth
        * height
        * width
    )

    data = [
        random.uniform(0, 1)
        for _ in range(total_values)
    ]

    payload = {
        "data": data,
        "channels": channels,
        "depth": depth,
        "height": height,
        "width": width,
    }

    with open(
        "api/test_payload.json",
        "w",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )

    print(
        f"Created payload with "
        f"{total_values} values."
    )


if __name__ == "__main__":
    main()