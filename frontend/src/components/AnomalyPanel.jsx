function AnomalyPanel({ predictions }) {
  const anomalyZones = predictions.filter(
    (prediction) => prediction.anomaly_level && prediction.anomaly_level !== "Low"
  );

  return (
    <div className="anomaly-section">
      <div className="anomaly-header">
        <h3>🧪 Spectral / Chemical Anomalies</h3>
        <p>Detected unusual crop signatures</p>
      </div>

      {anomalyZones.length === 0 ? (
        <p>No significant anomalies detected.</p>
      ) : (
        anomalyZones.map((prediction) => (
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
              {prediction.anomaly_type || "Unknown"}
            </p>

            <p>
              <strong>Spectral Deviation:</strong>{" "}
              {typeof prediction.spectral_deviation === "number"
                ? prediction.spectral_deviation.toFixed(2)
                : "N/A"}
            </p>

            <p>
              <strong>Risk Score:</strong>{" "}
              {typeof prediction.risk_score === "number"
                ? prediction.risk_score.toFixed(2)
                : "N/A"}
            </p>
          </div>
        ))
      )}
    </div>
  );
}

export default AnomalyPanel;