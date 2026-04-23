// tensorflowUtils.js

/**
 * Filter predictions by confidence score
 * @param {Array} predictions - Array of predictions from the model
 * @param {number} threshold - Confidence threshold (0-1)
 */
export function filterByConfidence(predictions, threshold) {
  if (!predictions) return [];
  return predictions.filter(p => p.score >= threshold);
}

/**
 * Count objects by class for the "Results" sidebar
 * @param {Array} predictions - Array of predictions
 */
export function getClassDistribution(predictions) {
  return predictions.reduce((acc, pred) => {
    acc[pred.class] = (acc[pred.class] || 0) + 1;
    return acc;
  }, {});
}