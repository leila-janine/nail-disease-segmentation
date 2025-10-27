import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import time # Optional: To simulate processing time if needed for testing spinner

# --- Page Configuration (Set Title and Layout) ---
st.set_page_config(page_title="Nail Disease Segmentation", layout="wide")

# --- Configuration ---
MODEL_PATH = "best.pt" # Make sure best.pt is in the same folder as app.py
CONFIDENCE_THRESHOLD = 0.20 # Keep low for detecting healthy/subtle cases
MASK_ALPHA = 0.5 # Transparency of the segmentation masks (0.0 to 1.0)
PROJECT_GROUP_NAME = "youngstunna" # Your group name

# --- Inject Custom CSS ---
# (Same CSS as provided in the previous response - included here for completeness)
st.markdown("""
<style>
/* --- General App Styling --- */
body {
    background-color: #f0f2f6; /* Light gray background */
}
/* Hide default Streamlit header and hamburger menu */
.stApp > header {
    background-color: transparent;
    display: none;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Main content area styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1200px; /* Limit max width */
    margin: auto;
    background-color: #e8edf5; /* Light blue-gray page background */
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* --- Column Styling --- */
/* Target the div containing the columns */
div[data-testid="stHorizontalBlock"] {
    /* Optional: Add styles if needed */
}

/* --- Left Panel Styling (Blue) --- */
/* Target first column's inner div directly */
div[data-testid="stVerticalBlock"]:nth-child(1) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background-color: #0047AB; /* Ximilar's blue */
    color: white;
    padding: 3rem;
    border-radius: 10px;
    height: 100%; /* Make panels equal height */
    box-sizing: border-box; /* Include padding in height calculation */
}
/* Force text color in the first column */
div[data-testid="stVerticalBlock"]:nth-child(1) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] * {
    color: white !important;
}
/* Style headers specifically */
div[data-testid="stVerticalBlock"]:nth-child(1) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] h1,
div[data-testid="stVerticalBlock"]:nth-child(1) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] h2,
div[data-testid="stVerticalBlock"]:nth-child(1) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] h3 {
     color: white !important;
     padding-bottom: 1rem;
}

/* --- Right Panel Styling (White) --- */
/* Target second column's inner div directly */
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background-color: white;
    padding: 3rem;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    height: 100%; /* Make panels equal height */
    box-sizing: border-box; /* Include padding in height calculation */
}

/* Style file uploader label in the second column */
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] .stFileUploader label {
    font-weight: bold;
    color: #333 !important; /* Force color */
    padding-bottom: 0.5rem;
}
/* Style file uploader border */
div[data-testid="stFileUploader"] section {
    border: 2px dashed #ccc;
    background-color: #f8f9fa;
    padding: 1.5rem; /* Increase padding */
    border-radius: 5px;
}
div[data-testid="stFileUploader"] section:hover {
     background-color: #e9ecef;
}
/* Style titles/subheaders in the second column */
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] h1,
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] h2,
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] h3 {
     color: #333 !important; /* Darker text */
     padding-bottom: 1rem;
}
/* Style general text in the second column */
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] div,
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] p,
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] li {
    color: #555 !important; /* Slightly lighter text */
}

/* Style the spinner text */
.stSpinner > div > div {
    color: #0047AB !important; /* Blue spinner text */
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# --- Model Loading (Cached) ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model."""
    try:
        model = YOLO(model_path)
        # Perform a dummy inference to ensure model loads correctly (optional)
        # model(np.zeros((640, 640, 3), dtype=np.uint8))
        return model
    except Exception as e:
        print(f"Error loading model: {e}") # Log error for debugging
        return None

# --- Define the two main columns ---
col1, col2 = st.columns([1, 1.2], gap="large") # Give slightly more space to right column

# --- Content for Left Column (Blue Area) ---
with col1:
    # st.image("your_logo.png", width=150) # Replace with your logo if you have one
    st.header("Nail Disease Segmentation")
    st.write(f"""
        An AI-powered application developed by **Group {PROJECT_GROUP_NAME}** for the AI2 T1 AY2526 course.
        This tool analyzes nail images to identify potential health conditions.
        Upload an image, and the system will attempt to segment and classify areas indicating specific nail diseases or confirm healthy nails.
    """)
    st.markdown("---")
    st.write("**How it works:**")
    st.write("1. Upload a clear image of a nail.")
    st.write("2. The AI model analyzes the image.")
    st.write("3. Detected conditions (or healthy status) are highlighted.")
    st.write("_Disclaimer: This tool is for educational purposes only and not a substitute for professional medical diagnosis._")

# --- Content for Right Column (White Area) ---
with col2:
    st.subheader("Analyze Your Image")

    model = load_yolo_model(MODEL_PATH)
    if model is None:
        st.error(f"FATAL ERROR: Model failed to load from path '{MODEL_PATH}'. Ensure 'best.pt' is in the application directory and not corrupted.")
        st.stop()
    else:
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"], label_visibility="visible")

        if uploaded_file is not None:
            # Create placeholders for results outside the spinner
            result_placeholder = st.empty()
            message_placeholder = st.empty()

            try:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                img_cv = np.array(image.convert('RGB'))
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

                result_placeholder.image(image, caption='Uploaded Image.', use_container_width=True)

                # Use st.spinner for the processing block
                with st.spinner("Analyzing image..."):
                    results = model(img_cv, conf=CONFIDENCE_THRESHOLD)

                    overlay_image = img_cv.copy()
                    detection_made = False
                    detected_classes = set()

                    names = model.names
                    colors = [tuple(np.random.randint(100, 256, 3).tolist()) for _ in range(len(names))]

                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()

                        if boxes.shape[0] > 0:
                            class_ids = r.boxes.cls.cpu().numpy().astype(int)
                            detection_made = True
                            for cls_id in class_ids:
                                 if 0 <= cls_id < len(names):
                                     detected_classes.add(names[cls_id])
                        else:
                            class_ids = np.array([], dtype=int)

                        # Segmentation Masks
                        if r.masks is not None and len(class_ids) > 0:
                            masks = r.masks.data.cpu().numpy()
                            overlay_h, overlay_w = overlay_image.shape[:2]

                            for i, mask in enumerate(masks):
                                if i < len(class_ids):
                                    mask_resized = cv2.resize(mask, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
                                    mask_uint8 = mask_resized.astype(np.uint8) * 255
                                    class_id = class_ids[i]
                                    if 0 <= class_id < len(colors):
                                        color = colors[class_id]
                                        colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
                                        for c_idx in range(3):
                                           colored_mask[:, :, c_idx] = np.where(mask_uint8 == 255, color[c_idx], 0)
                                        overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_mask, MASK_ALPHA, 0)

                        # Bounding Boxes and Labels
                        if class_ids.size > 0:
                            for i, (box, score) in enumerate(zip(boxes, scores)):
                                if i < len(class_ids):
                                   cls_id = class_ids[i]
                                   if 0 <= cls_id < len(names):
                                        x1, y1, x2, y2 = map(int, box)
                                        label = f"{names[cls_id]} {score:.2f}"
                                        color = colors[cls_id]
                                        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)
                                        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        y_text_bg_top = max(0, y1 - label_height - baseline - 5)
                                        cv2.rectangle(overlay_image, (x1, y_text_bg_top), (x1 + label_width, y1), color, cv2.FILLED)
                                        y_text_pos = max(10, y1 - baseline - 3)
                                        cv2.putText(overlay_image, label, (x1, y_text_pos),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                                    lineType=cv2.LINE_AA)

                # --- Refined Display Logic (Outside Spinner) ---
                if detection_made:
                    is_only_healthy = detected_classes == {'healthy_nail'}

                    if is_only_healthy:
                        message_placeholder.success("Healthy nail detected. No diseases found based on the analysis.")
                        result_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
                        # Update the result placeholder with the processed image
                        result_placeholder.image(result_image_rgb, caption='Processed Image - Healthy Nail.', use_container_width=True)
                    else:
                        # Diseases were detected (or a mix)
                        message_placeholder.warning("Nail condition(s) detected. This is not a medical diagnosis. Please consult a healthcare professional.")
                        result_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
                        # Update the result placeholder with the processed image
                        result_placeholder.image(result_image_rgb, caption='Processed Image with Detections.', use_container_width=True)

                else:
                    # Nothing detected above threshold
                    message_placeholder.info(f"No nail conditions (including healthy) were detected above the {CONFIDENCE_THRESHOLD*100:.0f}% confidence threshold.")
                    # Keep showing the original uploaded image in the result placeholder
                    result_placeholder.image(image, caption='Uploaded Image.', use_container_width=True)


            except Exception as e:
                # Clear placeholders on error and show error message
                result_placeholder.empty()
                message_placeholder.empty()
                st.error(f"An error occurred during image processing: {e}")
                st.warning("Please ensure you uploaded a valid, uncorrupted image file (JPG, PNG, JPEG).")

# --- Sidebar for Ethics Notice ---
# (Keep the sidebar code exactly the same as the previous response)
st.sidebar.title("Ethical Considerations")
st.sidebar.markdown("---")
st.sidebar.subheader("Notice on Use, Redistribution, and Ethical Compliance")
st.sidebar.warning("Redistribution, reproduction, or use of this material beyond personal reference is strictly prohibited without the prior written consent of the author. Unauthorized copying, modification, or dissemination—whether for commercial, academic, or institutional purposes—violates intellectual property rights and may result in legal or disciplinary action.")

st.sidebar.subheader("AI Governance and Ethics Considerations")
st.sidebar.error("This work must not be used in ways that:")
st.sidebar.markdown("""
* Compromise data privacy or violate data protection regulations (e.g., GDPR, Philippine Data Privacy Act).
* Perpetuate bias or discrimination by misusing algorithms, datasets, or results.
* Enable harmful applications, including surveillance, profiling, or uses that undermine human rights.
* Misrepresent authorship or credit, such as plagiarism or omission of proper citations.
""")

st.sidebar.subheader("Responsible Use Principles")
st.sidebar.info("Users are expected to follow responsible research and innovation practices, ensuring that any derivative work is:")
st.sidebar.markdown("""
* **Transparent** → Clear acknowledgment of sources and methodology.
* **Accountable** → Proper attribution and disclosure of limitations.
* **Beneficial to society** → Applications that align with ethical standards and do not cause harm.
""")
st.sidebar.markdown("---")
st.sidebar.caption("For any intended use (academic, research, or practical), prior written approval must be obtained from the author to ensure compliance with both legal requirements and ethical AI practices.")