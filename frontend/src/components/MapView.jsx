import Map from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";

import "mapbox-gl/dist/mapbox-gl.css";

import mockPredictions from "../data/mockPredictions";

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

function MapView() {
  const layers = [
    // Disease-risk heatmap
    new HeatmapLayer({
      id: "risk-heatmap",

      data: mockPredictions,

      getPosition: (d) => [
        d.longitude,
        d.latitude
      ],

      // Risk score controls heatmap intensity
      getWeight: (d) => d.risk_score,

      radiusPixels: 60,

      intensity: 1,

      threshold: 0.05
    }),

    // Individual prediction points
    new ScatterplotLayer({
      id: "prediction-points",

      data: mockPredictions,

      getPosition: (d) => [
        d.longitude,
        d.latitude
      ],

      getRadius: 30,

      getFillColor: (d) => {
        if (d.risk_score >= 0.7) {
          return [255, 0, 0];
        }

        if (d.risk_score >= 0.4) {
          return [255, 165, 0];
        }

        return [0, 200, 0];
      },

      pickable: true,

      radiusMinPixels: 8
    })
  ];

  return (
    <DeckGL
      initialViewState={{
        longitude: 77.752,
        latitude: 8.721,
        zoom: 14
      }}
      controller={true}
      layers={layers}
      getTooltip={({ object }) =>
  object
    ? {
        text:
          `Prediction: ${object.prediction_id}\n` +
          `Zone: ${object.zone_id}\n` +
          `Disease: ${object.disease}\n` +
          `Risk Score: ${object.risk_score}\n` +
          `Status: ${object.status}`
      }
    : null
}
    >
      <Map
        mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        mapboxAccessToken={MAPBOX_TOKEN}
      />
    </DeckGL>
  );
}

export default MapView;