import streamlit as st
from PIL import Image

from src.predict import DEFAULT_MODEL, PlateDetector

st.set_page_config(page_title="Car Plate Detection", page_icon="🚗")
st.title("Car Plate Detection 🚗📸")
st.caption("VGG16 backbone + a small regression head that predicts one bounding box per image.")


@st.cache_resource
def get_detector():
    return PlateDetector(DEFAULT_MODEL)


try:
    detector = get_detector()
except Exception:
    st.error(f"Couldn't load `{DEFAULT_MODEL}`. Train the model first: `python -m src.train --data-dir data`")
    st.stop()

uploaded = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    box = detector.predict_box(image)

    left, right = st.columns(2)
    left.image(image, caption="Original", use_container_width=True)
    right.image(detector.draw(image, box), caption="Detected plate", use_container_width=True)

    st.subheader("Cropped plate")
    st.image(detector.crop(image, box), width=300)
    st.caption(f"Box (xmin, ymin, xmax, ymax): {box}")
    st.info("This model always returns one box, even if there is no plate in the picture.")
