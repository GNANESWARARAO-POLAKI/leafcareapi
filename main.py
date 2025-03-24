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

# ✅ Load YOLO Model (for leaf detection)
leaf_model = YOLO("foduucom/plant-leaf-detection-and-classification")  
leaf_model.overrides['conf'] = 0.40  # Confidence threshold
leaf_model.overrides['iou'] = 0.90   # IoU threshold
leaf_model.overrides['agnostic_nms'] = True  # Class-agnostic NMS
leaf_model.overrides['max_det'] = 1000  # Max detections per image

# ✅ Load CNN Model (for disease classification)
cnn_model = tf.keras.models.load_model('./trained_plant_disease_model.keras')

# ✅ Get the input size expected by the CNN model
input_shape = cnn_model.input_shape  # Example: (None, 224, 224, 3)
target_size = (input_shape[1], input_shape[2])  # Extract required image size

# ✅ Class Names (Diseases)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

@app.get("/")
def home():
    return {"message": "Welcome to the Leaf Detection and Disease Classification API"}

@app.post("/detect_leaf_disease/")
async def detect_leaf_disease(file: UploadFile = File(...)):
    """Detects if a leaf is present, then classifies the disease."""
    try:
        # Step 1: Validate file type
        # if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        #     raise HTTPException(status_code=400, detail="Invalid file format. Use JPEG, JPG, or PNG.")

        # Step 2: Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image.write(await file.read())
            temp_image_path = temp_image.name  # Get temp file path
        
        print(f"✅ Image saved at: {temp_image_path}")

        # Step 3: Open image for processing
        image = Image.open(temp_image_path).convert("RGB")
        

        # Step 4: Detect if a leaf is present using YOLO
        
        leaf_results = leaf_model.predict(temp_image_path, verbose=False)
        filtered_boxes = [box for box in leaf_results[0].boxes if box.conf[0] > 0.5]
        leaf_detected = len(filtered_boxes)>0

        # Save the rendered image with detection results
        if leaf_detected:
            render = render_result(model=leaf_model, image=image, result=leaf_results[0])
            render_path = temp_image_path.replace(".jpg", "_detected.jpg")
            render.show()
            print(f"✅ Detection image saved at: {render_path}")

        if not leaf_detected:
            os.remove(temp_image_path)  # Cleanup
            return {
                "success": False,
                "message": "No leaf detected",
                "leaf_detected": False,
                "disease_detected": False
            }

        # Step 5: Classify Disease Using CNN Model
        disease_name, confidence = predict_disease(temp_image_path)

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

def predict_disease(image_path):
    """Predicts the plant disease using the CNN model."""
    try:
        # Load image and resize
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Convert image to array
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Get CNN prediction
        predictions = cnn_model.predict(input_arr)
        predicted_class = np.argmax(predictions)  # Get highest probability index
        confidence = np.max(predictions)  # Get confidence score

        return class_names[predicted_class], confidence

    except Exception as e:
        print(f"❌ Error in Disease Prediction: {e}")
        return None, None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
