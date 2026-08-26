import mockPredictions from "../data/mockPredictions";

export const fetchPredictions = async () => {
  // Temporary mock API simulation
  await new Promise((resolve) => setTimeout(resolve, 800));

  return mockPredictions;
};