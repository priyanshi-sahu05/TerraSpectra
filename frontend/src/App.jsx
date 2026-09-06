import { useEffect, useState } from "react";

import MapView from "./components/MapView";
import RiskSummary from "./components/RiskSummary";
import Loading from "./components/Loading";
import { fetchPredictions } from "./services/predictionService";
import Timeline from "./components/Timeline";
import historicalPredictions from "./data/historicalPredictions";

function App() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const dates = Object.keys(historicalPredictions);
  const [selectedDate, setSelectedDate] = useState(dates[0]);
  
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
    return <Loading/>;
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
      <header className="dashboard-header">
        <h1>TerraSpectra GIS Dashboard</h1>
        <p>Hyperspectral Crop Disease Monitoring</p>
      </header>

      <div className="dashboard-content">
        <main className="map-container">
          <MapView predictions={predictions} />
        </main>

        <RiskSummary predictions={predictions} />
      </div>

      <Timeline
        dates={dates}
        selectedDate={selectedDate}
        onDateChange={setSelectedDate}
      />
    </div>
  );
}
export default App;