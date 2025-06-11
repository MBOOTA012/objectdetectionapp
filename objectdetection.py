import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (change to your trained model path if needed)
model = YOLO('yolo11n.pt')  # or 'runs/detect/train/weights/best.pt'

# Streamlit UI
st.title("🧠 YOLOv8 Object Detection App")
st.markdown("Upload an image and detect objects using a trained YOLOv8 model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to NumPy array
    image_np = np.array(image)

    # Run detection
    st.write("🔍 Detecting objects...")
    results = model(image_np)

    # Plot result with bounding boxes
    result_image = results[0].plot()

    # Display result
    st.image(result_image, caption="Detected Objects", use_column_width=True)

    # Optional: Show raw detections
    with st.expander("🔎 Detection Details"):
        for box in results[0].boxes:
            st.write({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "coordinates": box.xyxy[0].tolist()
            })
