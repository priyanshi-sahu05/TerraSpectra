import mockPredictions from "../data/mockPredictions";

function RiskSummary() {
  const highRiskCount = mockPredictions.filter(
    (prediction) => prediction.risk_score >= 0.7
  ).length;

  const mediumRiskCount = mockPredictions.filter(
    (prediction) =>
      prediction.risk_score >= 0.4 &&
      prediction.risk_score < 0.7
  ).length;

  const lowRiskCount = mockPredictions.filter(
    (prediction) => prediction.risk_score < 0.4
  ).length;

  const totalZones = mockPredictions.length;

  return (
    <div className="risk-summary">
      <h2>Risk Summary</h2>

      <div className="risk-item">
        <span>🔴 High Risk</span>
        <strong>{highRiskCount}</strong>
      </div>

      <div className="risk-item">
        <span>🟠 Medium Risk</span>
        <strong>{mediumRiskCount}</strong>
      </div>

      <div className="risk-item">
        <span>🟢 Low Risk</span>
        <strong>{lowRiskCount}</strong>
      </div>

      <div className="risk-item total">
        <span>Total Zones</span>
        <strong>{totalZones}</strong>
      </div>
    </div>
  );
}

export default RiskSummary;