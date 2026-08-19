# TerraSpectra GIS Coordinate Mapping

## 1. Overview

TerraSpectra uses geographical coordinates to display crop disease prediction results on the GIS dashboard.

The prediction data contains latitude and longitude values.

The frontend uses React, Deck.gl, and Mapbox to visualize these coordinates.

---

## 2. Coordinate Structure

Each prediction contains:

- prediction_id
- zone_id
- latitude
- longitude
- risk_score
- status
- disease

Example:

```javascript
{
  prediction_id: "PRED_001",
  zone_id: "ZONE_01",
  latitude: 8.7205,
  longitude: 77.7505,
  risk_score: 0.25,
  status: "Low Risk",
  disease: "Healthy"
}