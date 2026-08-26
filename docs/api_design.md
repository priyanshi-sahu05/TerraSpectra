## Request Validation

The `/predict` endpoint accepts flattened numerical
input data together with tensor dimensions.

Example:

```json
{
  "data": [...],
  "channels": 1,
  "depth": 8,
  "height": 8,
  "width": 8
}