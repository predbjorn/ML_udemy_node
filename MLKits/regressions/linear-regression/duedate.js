require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const dayjs = require("dayjs");
const loadCSV = require("../load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

// ################
//   SECTION 6 - 8
// ################

let { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/duedateWIP.csv",
  {
    shuffle: true,
    splitTest: 20,
    dataColumns: ["mothersHeight", "maternalBmi"],
    labelColumns: ["days"],
  }
);

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100,
  batchSize: 10,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Mean Squared Error",
});

console.log("R2 is", r2);

const LMP = dayjs().subtract(4, "weeks");
console.log({ LMP: LMP.toDate() });
console.log(
  "Nørsi mener å vite hvor mange dager du skal legge til. Utgangspunktet er at du hadde siste mens for 4 uker siden."
);
const predictBasedOnBMIAndHeight = (height, bmi) => {
  const calculatedLMP = regression.predict([[height, bmi]]).get(0, 0);
  const restOfDay = calculatedLMP - Math.floor(calculatedLMP);
  const minutes = restOfDay * (60 * 24);

  console.log(
    `Mor er ${height}cm høy og har bmi på ${bmi}: Nørsi mener terminen er: ${LMP.add(
      calculatedLMP,
      "days"
    )
      .add(minutes, "minutes")
      .format("YYYY-MM-DD HH:mm")} (LMP + ${calculatedLMP})`
  );
};

predictBasedOnBMIAndHeight(162, 21.3);
predictBasedOnBMIAndHeight(172, 24);
predictBasedOnBMIAndHeight(177, 26);
predictBasedOnBMIAndHeight(152, 23);
