# import os
import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tempfile
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
# from io import BytesIO
from PIL import Image
from ultralyticsplus import YOLO, render_result
import uvicorn

# Initialize FastAPI
app = FastAPI()


# ✅ Load TFLite model
interpreter = tf.lite.Interpreter(model_path="./model/model.tflite")
interpreter.allocate_tensors()

# ✅ Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Get input shape
input_shape = input_details[0]['shape']
target_size = (input_shape[1], input_shape[2])  # e.g., (224, 224)


# ✅ Load YOLO Model (for leaf detection)
leaf_model = YOLO("foduucom/plant-leaf-detection-and-classification")
leaf_model.overrides['conf'] = 0.40  # Confidence threshold
leaf_model.overrides['iou'] = 0.90   # IoU threshold
leaf_model.overrides['agnostic_nms'] = True  # Class-agnostic NMS
leaf_model.overrides['max_det'] = 1000  # Max detections per image

# ✅ Load CNN Model (for disease classification)
# cnn_model = tf.keras.models.load_model("model/rained_plant_disease_model.keras" )

# ✅ Get the input size expected by the CNN model
# input_shape = cnn_model.input_shape  # Example: (None, 224, 224, 3)
# target_size = (input_shape[1], input_shape[2])  # Extract required image size

# ✅ Class Names (Diseases)
class_names = [
    'Apple Apple_scab', 'Apple Black_rot', 'Apple Cedar_apple_rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry_(including_sour) Powdery_mildew', 'Cherry_(including_sour) healthy',
    'Corn_(maize) Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize) Common_rust_',
    'Corn_(maize) Northern_Leaf_Blight', 'Corn_(maize) healthy', 'Grape Black_rot',
    'Grape Esca_(Black_Measles)', 'Grape Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape healthy',
    'Orange Haunglongbing_(Citrus_greening)', 'Peach Bacterial_spot', 'Peach healthy',
    'Pepper,_bell Bacterial_spot', 'Pepper,_bell healthy', 'Potato Early_blight',
    'Potato Late_blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
    'Squash Powdery_mildew', 'Strawberry Leaf_scorch', 'Strawberry healthy',
    'Tomato Bacterial_spot', 'Tomato Early_blight', 'Tomato Late_blight', 'Tomato Leaf_Mold',
    'Tomato Septoria_leaf_spot', 'Tomato Spider_mites Two-spotted_spider_mite', 'Tomato Target_Spot',
    'Tomato Tomato_Yellow_Leaf_Curl_Virus', 'Tomato Tomato_mosaic_virus', 'Tomato healthy'
]

plants=['apple','potato','tomato','corn','Blueberry','strawberry','soyabean','peach','grape','cherry','raspberry','orange','pepper','squash']
# Extract labels

@app.get("/")
def home():
    return {"message": "Welcome to the Leaf Detection and Disease Classification API"}

@app.post("/detect_leaf_disease/")
async def detect_leaf_disease(file: UploadFile = File(...)):
    """Detects if a leaf is present, then classifies the disease."""
    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image.write(await file.read())
            temp_image_path = temp_image.name  # Get temp file path

        print(f"✅ Image saved at: {temp_image_path}")
        image = Image.open(temp_image_path).convert("RGB")

        leaf_results = leaf_model.predict(temp_image_path, verbose=False)
        filtered_boxes = [leaf_model.names[int(box.cls)] for box in leaf_results[0].boxes if (box.conf > 0.2 and leaf_model.names[int(box.cls)] in plants)]
        print(filtered_boxes)
        leaf_detected = len(filtered_boxes)>0

        if leaf_detected:
            render = render_result(model=leaf_model, image=image, result=leaf_results[0])
            render_path = temp_image_path.replace(".jpg", "_detected.jpg")
            # render.show()
            # print(f"✅ Detection image saved at: {render_path}")

        if not leaf_detected:
            os.remove(temp_image_path)  # Cleanup
            return {
                "success": False,
                "message": "No leaf detected",
                "leaf_detected": False,
                "disease_detected": False
            }


        # Step 5: Classify Disease Using CNN Model
        disease_name, confidence = predict_disease(temp_image_path,filtered_boxes)

        # Cleanup temp image
        os.remove(temp_image_path)

        if not disease_name:
            return {
                "success": True,
                "message": "Leaf detected, but no disease found",
                "leaf_detected": True,
                "disease_detected": False,
            }

        return {
            "success": True,
            "message": "Leaf detected, disease found",
            "leaf_detected": True,
            "disease_detected": True,
            "disease_name": disease_name,
            "confidence": f"{confidence}",
            "detection_image_path": render_path  # Path to the saved detected image
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

from PIL import Image
import numpy as np

def predict_disease(image_path, filtered_boxes):
    """Predicts the plant disease using the TFLite model, without tf.keras preprocessing."""
    try:
        # ✅ Load and preprocess image using PIL
        image = Image.open(image_path).convert("RGB")
        image = image.resize(target_size)  # Resize to match model input
        input_arr = np.array(image).astype(np.float32)  # Convert to NumPy array and float32

        # ✅ Normalize if needed (comment/uncomment depending on how model was trained)
        # input_arr = input_arr / 255.0

        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Set tensor to interpreter
        interpreter.set_tensor(input_details[0]['index'], input_arr)
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        confidence = np.max(output_data)

        return class_names[predicted_class], confidence

    except Exception as e:
        print(f"❌ Error in TFLite Prediction: {e}")
        return None, None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)