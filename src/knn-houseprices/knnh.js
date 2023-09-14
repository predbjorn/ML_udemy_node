require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const k = 2;
const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7],
]);

const labels = tf.tensor([[200], [250], [215], [240]]);

const predictionPoint = tf.tensor([-121, 47]);

const distances = features
  .sub(predictionPoint)
  .pow(2)
  .sum(1)
  .pow(0.5)
  .expandDims(1);

const indices = distances
  .concat(labels, 1)
  .unstack()
  .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
  .slice(0, k);

const result = indices.reduce((acc, pair) => acc + pair.get(1), 0) / k;
console.log("Guess", result);


e rreefrere