const tf = require("@tensorflow/tfjs");

// Crear el modelo
const model = tf.sequential();

// Agregar una capa densa con más neuronas y activación ReLU
model.add(tf.layers.dense({ units: 64, inputShape: [2], activation: "relu" }));
model.add(tf.layers.dense({ units: 1 }));

// Compilar el modelo con optimizador Adam
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(0.001),
});

// Datos de entrenamiento (más complejos)
const xs = tf.tensor2d(
  [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
  ],
  [4, 2]
);
const ys = tf.tensor2d([3, 5, 7, 9], [4, 1]);

// Entrenar el modelo
async function trainModel() {
  await model.fit(xs, ys, { epochs: 500 });
  console.log("Entrenamiento completado.");
}

// Predicción
async function predictValue(x) {
  const prediction = model.predict(tf.tensor2d([[x, x + 1]], [1, 2]));
  prediction.print();
}

// Entrenamiento y predicción
trainModel().then(() => predictValue(5));
