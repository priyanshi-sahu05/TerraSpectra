# TerraSpectra GIS Dashboard

## Overview

The TerraSpectra GIS Dashboard is the frontend visualization module of the project.

It displays crop disease prediction results on an interactive map.

## Technologies

### React

React is used to build the dashboard user interface.

### Mapbox

Mapbox provides the interactive geographical map.

### Deck.gl

Deck.gl is used to visualize prediction data on top of the map.

### ScatterplotLayer

ScatterplotLayer displays individual prediction zones as points.

### HeatmapLayer

HeatmapLayer visualizes areas with higher disease-risk concentration.

### Axios

Axios is used for communication with the backend prediction API.

## Coordinate Mapping

Each prediction contains:

- latitude
- longitude
- risk_score
- zone_id
- status

Example:

```javascript
{
  zone_id: "ZONE_03",
  latitude: 8.7195,
  longitude: 77.7535,
  risk_score: 0.87,
  status: "High Risk"
}