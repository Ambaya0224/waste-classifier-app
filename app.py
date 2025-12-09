import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================================
# CONFIG
# ==================================================
MODEL_PATH = "best_mobilenetv2.keras"   # use the best model saved during training
IMG_SIZE = 224
CLASS_NAMES = ["Organic", "Recyclable"]  # adjust if needed


# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


# ==================================================
# PREPROCESS FUNCTION
# ==================================================
def preprocess(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # make batch of 1
    return image


# ==================================================
# STREAMLIT UI
# ==================================================
st.title("♻️ Waste Classifier – Organic vs Recyclable")
st.write("Upload an image of waste to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            input_tensor = preprocess(image)
            preds = model.predict(input_tensor)[0]

            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.success(f"### 🟢 Prediction: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

            st.write("Raw prediction probabilities:")
            st.json({CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))})

