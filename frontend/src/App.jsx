import MapView from "./components/MapView";
import RiskSummary from "./components/RiskSummary";

function App() {
  return (
    <div className="dashboard">
      <h1>TerraSpectra GIS Dashboard</h1>

      <div className="dashboard-content">
        <div className="map-container">
          <MapView />
        </div>

        <RiskSummary />
      </div>
    </div>
  );
}

export default App;