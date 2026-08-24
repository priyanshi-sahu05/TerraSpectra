import mockPredictions from "../data/mockPredictions";

function AnomalyPanel() {
  const anomalyZones = mockPredictions.filter(
    (prediction) => prediction.anomaly_level !== "Low"
  );

  return (
    <div className="anomaly-section">
      <div className="anomaly-header">
        <h3>🧪 Spectral / Chemical Anomalies</h3>
        <p>Detected unusual crop signatures</p>
      </div>

      {anomalyZones.map((prediction) => (
        <div
          className="anomaly-card"
          key={prediction.zone_id}
        >
          <div className="anomaly-card-header">
            <strong>{prediction.zone_id}</strong>

            <span
              className={`anomaly-level ${prediction.anomaly_level.toLowerCase()}`}
            >
              {prediction.anomaly_level}
            </span>
          </div>

          <p>
            <strong>Type:</strong>{" "}
            {prediction.anomaly_type}
          </p>

          <p>
            <strong>Spectral Deviation:</strong>{" "}
            {prediction.spectral_deviation.toFixed(2)}
          </p>

          <p>
            <strong>Risk Score:</strong>{" "}
            {prediction.risk_score.toFixed(2)}
          </p>
        </div>
      ))}
    </div>
  );
}

export default AnomalyPanel;