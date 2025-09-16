import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO
import io

# Initialize FastAPI app
app = FastAPI(title="YOLOv8 Model API", description="API for object detection using a custom YOLOv8 model.")

# Load the custom trained model
try:
    model = YOLO("Lycheeadd_trained_yolov8_model.pt")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

@app.get("/")
def home():
    """A simple welcome message to verify the API is running."""
    return {"message": "Welcome to the YOLOv8 API! Use the /predict endpoint to upload an image for detection."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts objects in an uploaded image using the YOLOv8 model.
    The endpoint accepts a file and returns the detection results.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # Read the image data from the uploaded file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Perform inference on the image
        results = model(image)
        
        # Process and return the results
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [float(val) for val in box.xyxy[0]]
                })
        
        return {"filename": file.filename, "detections": detections}
    except Exception as e:
        return {"error": str(e)}

# This part is for local testing. It won't be used on Render
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
