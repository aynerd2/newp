const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs');

const fs = require('fs');
const path = require('path');

const app = express();
const port = 8000;
const upload = multer({ dest: 'uploads/' });

// Path to your model file
const modelPath = path.join(__dirname, 'model', 'my_inception.json');

// Load TensorFlow.js model
async function loadModel() {
  try {
    const modelJSON = require(`${modelPath}`);
    const model = await tf.loadLayersModel(modelJSON);
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    return null;
  }
}

// Load model
let model;
async function init() {
  model = await loadModel();
}

init();

// Endpoint for image classification
app.post('/predict', upload.single('image'), async (req, res) => {
  if (!model) {
    return res.status(500).send('Model not loaded yet');
  }
  try {
    const imagePath = path.join(__dirname, req.file.path);
    const imageBuffer = fs.readFileSync(imagePath);
    const imageTensor = tf.node.decodeImage(imageBuffer);
    const inputTensor = imageTensor.expandDims(0).toFloat().div(tf.scalar(255));
    const prediction = await model.predict(inputTensor);
    const predictionValue = await prediction.data();
    res.send({ prediction: predictionValue });

    // Remove the uploaded image
    fs.unlinkSync(imagePath);
  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).send('Error during prediction');
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
