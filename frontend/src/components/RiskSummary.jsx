import mockPredictions from "../data/mockPredictions";
import AnomalyPanel from "./AnomalyPanel";

function RiskSummary() {
  const totalZones = mockPredictions.length;

  // Risk counts
  const highRiskCount = mockPredictions.filter(
    (prediction) => prediction.risk_score >= 0.7,
  ).length;

  const mediumRiskCount = mockPredictions.filter(
    (prediction) => prediction.risk_score >= 0.4 && prediction.risk_score < 0.7,
  ).length;

  const lowRiskCount = mockPredictions.filter(
    (prediction) => prediction.risk_score < 0.4,
  ).length;

  // Average risk score
  const averageRisk =
    mockPredictions.reduce(
      (total, prediction) => total + prediction.risk_score,
      0,
    ) / totalZones;

  // Highest risk prediction
  const highestRiskPrediction = mockPredictions.reduce(
    (highest, prediction) =>
      prediction.risk_score > highest.risk_score ? prediction : highest,
    mockPredictions[0],
  );

  // Acreage calculations
  const highRiskAcreage = mockPredictions
    .filter((prediction) => prediction.risk_score >= 0.7)
    .reduce((total, prediction) => total + prediction.acreage, 0);

  const mediumRiskAcreage = mockPredictions
    .filter(
      (prediction) =>
        prediction.risk_score >= 0.4 && prediction.risk_score < 0.7,
    )
    .reduce((total, prediction) => total + prediction.acreage, 0);

  const totalAcreageAtRisk = highRiskAcreage + mediumRiskAcreage;

  // Risk distribution percentages
  const highPercentage = (highRiskCount / totalZones) * 100;

  const mediumPercentage = (mediumRiskCount / totalZones) * 100;

  const lowPercentage = (lowRiskCount / totalZones) * 100;

  return (
    <aside className="analytics-sidebar">
      {/* Analytics Header */}
      <div className="analytics-header">
        <h2>Analytics</h2>
        <p>Crop Disease Risk Overview</p>
      </div>

      {/* Total Zones */}
      <div className="analytics-card">
        <span className="card-label">Total Prediction Zones</span>

        <strong className="card-value">{totalZones}</strong>
      </div>

      {/* Risk Counts */}
      <div className="risk-section">
        <h3>Risk Distribution</h3>

        <div className="risk-card high-risk">
          <span>🔴 High Risk</span>
          <strong>{highRiskCount}</strong>
        </div>

        <div className="risk-card medium-risk">
          <span>🟠 Medium Risk</span>
          <strong>{mediumRiskCount}</strong>
        </div>

        <div className="risk-card low-risk">
          <span>🟢 Low Risk</span>
          <strong>{lowRiskCount}</strong>
        </div>
      </div>

      {/* Distribution Bar */}
      <div className="distribution-section">
        <h3>Risk Distribution %</h3>

        <div className="distribution-bar">
          <div
            className="distribution-high"
            style={{
              width: `${highPercentage}%`,
            }}
          />

          <div
            className="distribution-medium"
            style={{
              width: `${mediumPercentage}%`,
            }}
          />

          <div
            className="distribution-low"
            style={{
              width: `${lowPercentage}%`,
            }}
          />
        </div>
      </div>

      {/* Acreage at Risk */}
      <div className="acreage-section">
        <h3>🌾 Acreage at Risk</h3>

        <div className="acreage-card high-acreage">
          <span>High Risk Area</span>
          <strong>{highRiskAcreage} acres</strong>
        </div>

        <div className="acreage-card medium-acreage">
          <span>Medium Risk Area</span>
          <strong>{mediumRiskAcreage} acres</strong>
        </div>

        <div className="acreage-total">
          <span>Total Area at Risk</span>
          <strong>{totalAcreageAtRisk} acres</strong>
        </div>
      </div>

      {/* Average Risk */}
      <div className="analytics-card">
        <span className="card-label">Average Risk Score</span>

        <strong className="card-value">
          {(averageRisk * 100).toFixed(1)}%
        </strong>
      </div>

      {/* Risk Score Visualization */}
      <div className="risk-score-section">
        <h3>Risk Score Visualization</h3>

        <div className="overall-risk">
          <span>Overall Risk Score</span>

          <strong>{(averageRisk * 100).toFixed(1)}%</strong>
        </div>

        <div className="risk-scale">
          <span>0</span>

          <div className="risk-scale-bar">
            <div
              className="risk-scale-indicator"
              style={{
                left: `${averageRisk * 100}%`,
              }}
            />
          </div>

          <span>1</span>
        </div>

        <div className="zone-risk-list">
          <h4>Zone Risk Scores</h4>

          {mockPredictions
            .slice()
            .sort((a, b) => b.risk_score - a.risk_score)
            .map((prediction) => (
              <div className="zone-risk-item" key={prediction.zone_id}>
                <div className="zone-risk-header">
                  <span>{prediction.zone_id}</span>

                  <strong>{prediction.risk_score.toFixed(2)}</strong>
                </div>

                <div className="zone-risk-bar">
                  <div
                    className="zone-risk-fill"
                    style={{
                      width: `${prediction.risk_score * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Anomaly Information */}

      <AnomalyPanel />

      {/* Highest Risk Zone */}
      <div className="highest-risk-card">
        <h3>Highest Risk Zone</h3>

        <p>
          <strong>{highestRiskPrediction.zone_id}</strong>
        </p>

        <p>
          Risk Score: <strong>{highestRiskPrediction.risk_score}</strong>
        </p>

        <p>
          Status: <strong>{highestRiskPrediction.status}</strong>
        </p>
      </div>
    </aside>
  );
}

export default RiskSummary;
