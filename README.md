# LeafCareAPI

LeafCareAPI is a FastAPI-based web service for plant leaf detection and disease classification using a TensorFlow Lite model. This API lets you upload plant leaf images and returns predictions about leaf presence and disease type.

## How to Run

### 1. Prerequisites

- **Python 3.9**
- The required model files in the `model/` directory, especially `model.tflite`
- Install dependencies from `requirements.txt`

### 2. Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

This launches the FastAPI server at [http://localhost:8000](http://localhost:8000).

### 3. API Endpoints

- **GET /**  
  Returns a welcome message.

- **POST /detect_leaf_disease/**  
  Upload a leaf image and get a prediction.  
  Example request using `curl`:
  ```bash
  curl -X POST "http://localhost:8000/detect_leaf_disease/" -F "file=@your_leaf.jpg"
  ```

## Key Code Explanation

### FastAPI Initialization

```python
app = FastAPI()
```
Creates the web API server.

### Model Loading

```python
interpreter = tf.lite.Interpreter(model_path="./model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```
Loads the TensorFlow Lite model for disease classification.

### Disease Classes

`class_names` holds all possible disease predictions (e.g., Apple Scab, Tomato Late Blight).

### Main Endpoint Logic

The `/detect_leaf_disease/` endpoint accepts image uploads:

```python
@app.post("/detect_leaf_disease/")
async def detect_leaf_disease(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    # Detect if a leaf is present (currently always true)
    # Predict disease using the TFLite model
```

- **Image Handling:**  
  Uploaded images are saved to a temporary file.
- **Prediction:**  
  The `predict_disease(image_path)` function:
  - Loads and preprocesses the image
  - Runs inference using the TFLite interpreter
  - Returns the most likely disease class and confidence score

### Example Prediction Flow

```python
def predict_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    input_arr = np.array(image).astype(np.float32)
    input_arr = np.expand_dims(input_arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)
    return class_names[predicted_class], confidence
```

## Notes

- The YOLO leaf detection code is present but commented out; currently, leaf detection is assumed.
- Make sure your `model.tflite` file is in the correct location.

---

Feel free to ask for more detail or section customization!