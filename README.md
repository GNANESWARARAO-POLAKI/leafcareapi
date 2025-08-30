# Leaf Care API 🌿

A powerful plant disease detection and classification API built with FastAPI and TensorFlow Lite. This API can identify and classify diseases in plant leaves across 14 different plant species.

## 🚀 Features

- **Plant Disease Detection**: Identifies diseases in uploaded leaf images
- **Multi-Plant Support**: Supports 14 different plant types
- **39 Disease Classifications**: Can detect and classify 39 different plant diseases
- **Fast Inference**: Uses TensorFlow Lite for optimized model performance
- **RESTful API**: Easy-to-use FastAPI endpoints
- **Docker Support**: Containerized deployment ready

## 🌱 Supported Plants

The API supports disease detection for the following plants:
- Apple
- Potato  
- Tomato
- Corn (Maize)
- Blueberry
- Strawberry
- Soybean
- Peach
- Grape
- Cherry
- Raspberry
- Orange
- Pepper (Bell)
- Squash

## 🦠 Supported Disease Classifications

The model can detect 39 different diseases and healthy states:

### Apple
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

### Blueberry
- Healthy

### Cherry
- Powdery Mildew
- Healthy

### Corn (Maize)
- Cercospora Leaf Spot (Gray Leaf Spot)
- Common Rust
- Northern Leaf Blight
- Healthy

### Grape
- Black Rot
- Esca (Black Measles)
- Leaf Blight (Isariopsis Leaf Spot)
- Healthy

### Orange
- Huanglongbing (Citrus Greening)

### Peach
- Bacterial Spot
- Healthy

### Pepper (Bell)
- Bacterial Spot
- Healthy

### Potato
- Early Blight
- Late Blight
- Healthy

### Raspberry
- Healthy

### Soybean
- Healthy

### Squash
- Powdery Mildew

### Strawberry
- Leaf Scorch
- Healthy

### Tomato
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted Spider Mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

## 🛠️ Model Architecture

The API uses a TensorFlow Lite model for efficient inference:

- **Model Type**: Convolutional Neural Network (CNN) optimized for mobile/edge deployment
- **Input Size**: 224x224 pixels RGB images
- **Framework**: TensorFlow Lite for fast inference
- **Preprocessing**: PIL-based image resizing and normalization
- **Output**: Disease classification with confidence score

### Model Pipeline
1. **Image Upload**: Accept image file via HTTP POST
2. **Preprocessing**: Resize image to 224x224, convert to RGB, normalize pixel values
3. **Inference**: Run through TensorFlow Lite CNN model
4. **Classification**: Return predicted disease class with confidence score

## 📋 Requirements

- Python 3.9+
- TensorFlow 2.19+
- FastAPI
- Pillow (PIL)
- NumPy
- Uvicorn

## 🚀 Quick Start

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/GNANESWARARAO-POLAKI/leafcareapi.git
cd leafcareapi
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the API:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. **Access the API:**
   - API Base URL: `http://localhost:8000`
   - Interactive Docs: `http://localhost:8000/docs`
   - Alternative Docs: `http://localhost:8000/redoc`

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t leafcareapi .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 leafcareapi
```

## 📝 API Usage

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response:**
```json
{
    "message": "Welcome to the Leaf Detection and Disease Classification API"
}
```

#### 2. Disease Detection
```http
POST /detect_leaf_disease/
```

**Parameters:**
- `file`: Image file (JPG, PNG, etc.)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/detect_leaf_disease/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/leaf_image.jpg"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/detect_leaf_disease/"
files = {"file": open("leaf_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Successful Response:**
```json
{
    "success": true,
    "message": "Leaf detected, disease found",
    "leaf_detected": true,
    "disease_detected": true,
    "disease_name": "Tomato Early_blight",
    "confidence": "0.8945",
    "detection_image_path": "null"
}
```

**No Disease Detected Response:**
```json
{
    "success": true,
    "message": "Leaf detected, but no disease found",
    "leaf_detected": true,
    "disease_detected": false
}
```

**Error Response:**
```json
{
    "success": false,
    "message": "No leaf detected",
    "leaf_detected": false,
    "disease_detected": false
}
```

## 🖼️ Image Requirements

For best results, ensure your leaf images meet these criteria:

- **Format**: JPG, PNG, or other common image formats
- **Quality**: Clear, well-lit images
- **Subject**: Single leaf or plant part with disease symptoms visible
- **Size**: Any size (will be automatically resized to 224x224)
- **Background**: Preferably plain or natural background

## 🔧 Development

### Project Structure
```
leafcareapi/
├── main.py                    # FastAPI application
├── model/
│   ├── model.tflite          # TensorFlow Lite model
│   └── trained_plant_disease_model.keras  # Original Keras model
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
└── README.md               # This file
```

### Running in Development Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-restart when code changes are detected.

## 🐛 Troubleshooting

### Common Issues

1. **TensorFlow Warnings**: CUDA/GPU warnings are normal and don't affect functionality
2. **Memory Issues**: Ensure sufficient RAM for model loading
3. **Port Conflicts**: Change port if 8000 is already in use: `--port 8001`
4. **Docker SSL Issues**: If Docker build fails with SSL certificate errors, try building with `--build-arg PIP_TRUSTED_HOST=pypi.org`

### Error Messages

- **"No leaf detected"**: Image doesn't contain recognizable plant material
- **"Error processing image"**: Invalid image format or corrupted file
- **Model loading errors**: Check if `model/model.tflite` exists
- **Docker build errors**: May occur in certain environments due to SSL certificates or package availability

## 📊 Performance

- **Inference Speed**: ~100-500ms per image (CPU)
- **Model Size**: Optimized TensorFlow Lite model
- **Memory Usage**: ~200-500MB RAM
- **Supported Formats**: JPG, PNG, JPEG, etc.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- TensorFlow team for the machine learning framework
- FastAPI for the excellent web framework
- Plant disease dataset contributors

## 📞 Support

For support, issues, or questions:
- Open an issue on GitHub
- Check the API documentation at `/docs` endpoint

---

**Happy Plant Health Monitoring! 🌿🏥**