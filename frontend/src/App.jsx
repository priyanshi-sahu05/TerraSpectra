import { useEffect, useState } from "react";

import MapView from "./components/MapView";
import RiskSummary from "./components/RiskSummary";

import { fetchPredictions } from "./services/predictionService";

function App() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadPredictions = async () => {
      try {
        setLoading(true);
        setError(null);

        const data = await fetchPredictions();

        setPredictions(data);
      } catch (err) {
        console.error("Failed to load predictions:", err);
        setError("Unable to load prediction data.");
      } finally {
        setLoading(false);
      }
    };

    loadPredictions();
  }, []);

  if (loading) {
    return (
      <div className="dashboard-status">
        <h2>Loading TerraSpectra predictions...</h2>
        <p>Please wait while prediction data is loading.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-status error">
        <h2>Prediction Data Error</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <h1>TerraSpectra GIS Dashboard</h1>

      <div className="dashboard-content">
        <div className="map-container">
          <MapView predictions={predictions} />
        </div>

        <RiskSummary predictions={predictions} />
      </div>
    </div>
  );
}

export default App;