const mockPredictions = [
  {
    prediction_id: "PRED_001",
    zone_id: "ZONE_01",
    latitude: 8.7205,
    longitude: 77.7505,
    risk_score: 0.25,
    status: "Low Risk",
    disease: "Healthy",
    acreage: 10
  },

  {
    prediction_id: "PRED_002",
    zone_id: "ZONE_02",
    latitude: 8.7215,
    longitude: 77.7520,
    risk_score: 0.52,
    status: "Medium Risk",
    disease: "Early Crop Stress",
    acreage: 12
  },

  {
    prediction_id: "PRED_003",
    zone_id: "ZONE_03",
    latitude: 8.7195,
    longitude: 77.7535,
    risk_score: 0.87,
    status: "High Risk",
    disease: "Possible Fungal Stress",
    acreage: 15
  },

  {
    prediction_id: "PRED_004",
    zone_id: "ZONE_04",
    latitude: 8.7230,
    longitude: 77.7515,
    risk_score: 0.91,
    status: "High Risk",
    disease: "Possible Fungal Stress",
    acreage: 20
  },

  {
    prediction_id: "PRED_005",
    zone_id: "ZONE_05",
    latitude: 8.7185,
    longitude: 77.7545,
    risk_score: 0.68,
    status: "Medium Risk",
    disease: "Moderate Crop Stress",
    acreage: 8
  }
];

export default mockPredictions;