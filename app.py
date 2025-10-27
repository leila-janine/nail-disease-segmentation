import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# --- Configuration ---
MODEL_PATH = "best.pt" # Make sure best.pt is in the same folder as app.py
CONFIDENCE_THRESHOLD = 0.25 # Adjust as needed (lower = more detections)
MASK_ALPHA = 0.5 # Transparency of the segmentation masks (0.0 to 1.0)

# --- Model Loading (Cached) ---
# Use st.cache_resource to load the model only once, improving performance.
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main App Logic ---
st.title("Nail Disease Segmentation")
st.write("Upload an image to detect nail conditions.")

# Load the model
model = load_yolo_model(MODEL_PATH)

if model is not None:
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try: # <<<<<<<< START OF TRY BLOCK >>>>>>>>>
            # Read the uploaded image file
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))

            # Convert PIL Image to OpenCV format (BGR)
            img_cv = np.array(image.convert('RGB'))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            # Display uploaded image (Use updated parameter)
            st.image(image, caption='Uploaded Image.', use_container_width=True) # FIX: use_container_width
            st.write("Processing...")

            # --- Run Inference ---
            results = model(img_cv, conf=CONFIDENCE_THRESHOLD) # Pass confidence threshold

            # --- Process and Visualize Results ---
            overlay_image = img_cv.copy() # Create a copy to draw on
            detected = False # Flag to check if anything was detected

            # Get class names from the model ONCE
            names = model.names
            # Generate distinct colors for each class ONCE
            colors = [tuple(np.random.randint(100, 256, 3).tolist()) for _ in range(len(names))]

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()

                # --- Define class_ids from boxes BEFORE loops, handling empty detections ---
                if boxes.shape[0] > 0: # Check if there are any boxes detected
                    class_ids = r.boxes.cls.cpu().numpy().astype(int) # FIX: Define class_ids from boxes
                else:
                    class_ids = np.array([], dtype=int) # Create empty array if no boxes/masks

                # --- Segmentation Masks (Draw first so boxes are on top) ---
                if r.masks is not None and len(class_ids) > 0: # Check if masks exist AND we have class IDs
                    detected = True # Mark detected if masks are present
                    masks = r.masks.data.cpu().numpy()
                    overlay_h, overlay_w = overlay_image.shape[:2] # Get image dimensions for resizing

                    for i, mask in enumerate(masks):
                        if i < len(class_ids): # Ensure mask index is valid for class_ids
                            # --- FIX: Resize mask to match image ---
                            mask_resized = cv2.resize(mask, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
                            mask_uint8 = mask_resized.astype(np.uint8) * 255 # Use the RESIZED mask

                            class_id = class_ids[i] # Get the class ID corresponding to this mask index
                            # Ensure class_id is within the valid range for colors
                            if 0 <= class_id < len(colors):
                                color = colors[class_id] # Get color for the current class

                                # Create colored mask overlay
                                colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
                                for c_idx in range(3): # Apply color channel by channel
                                   # Use the RESIZED mask (mask_uint8)
                                   colored_mask[:, :, c_idx] = np.where(mask_uint8 == 255, color[c_idx], 0)

                                # Blend the mask with the image
                                overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_mask, MASK_ALPHA, 0)
                            else:
                                print(f"Warning: Invalid class_id {class_id} encountered for mask index {i}. Skipping.")


                # --- Bounding Boxes and Labels ---
                # Check if class_ids has been defined and has items (already done above)
                if class_ids.size > 0:
                    for i, (box, score) in enumerate(zip(boxes, scores)): # Use enumerate if needed or just loop
                        if i < len(class_ids): # Ensure index is valid for class_ids
                           detected = True # Mark as detected if we draw boxes too
                           cls_id = class_ids[i] # Get class_id using the index

                           # Ensure class_id is within the valid range for colors and names
                           if 0 <= cls_id < len(names):
                                x1, y1, x2, y2 = map(int, box)
                                label = f"{names[cls_id]} {score:.2f}" # e.g., "pitting 0.85"
                                color = colors[cls_id] # Use the same color as the mask

                                # Draw rectangle
                                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)

                                # Put label + score above the box
                                # Calculate text size for background rectangle
                                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                # Draw background rectangle, ensure top coordinate is not negative
                                y_text_bg_top = max(0, y1 - label_height - baseline - 5)
                                cv2.rectangle(overlay_image, (x1, y_text_bg_top), (x1 + label_width, y1), color, cv2.FILLED)
                                # Put white text, ensure position is not negative
                                y_text_pos = max(10, y1 - baseline - 3) # Adjust y position slightly if near top
                                cv2.putText(overlay_image, label, (x1, y_text_pos),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                            lineType=cv2.LINE_AA)
                           else:
                               print(f"Warning: Invalid cls_id {cls_id} encountered for box index {i}. Skipping label.")

            # Display the result
            if detected:
                st.write("Detection Results:")
                # Convert BGR image (OpenCV) back to RGB for display in Streamlit
                result_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
                # Display processed image (Use updated parameter)
                st.image(result_image_rgb, caption='Processed Image with Detections.', use_container_width=True) # FIX: use_container_width
            else:
                st.warning("No objects detected with the current confidence threshold.")

        except Exception as e: # <<<<<<<< START OF EXCEPT BLOCK (Correctly indented) >>>>>>>>>
            # Graceful error handling for invalid files or processing errors
            st.error(f"An error occurred: {e}")
            st.warning("Please upload a valid image file (JPG, PNG, JPEG). If the error persists, the model might have issues processing this specific image.")
        # <<<<<<<< END OF EXCEPT BLOCK >>>>>>>>>

else:
    st.warning("Model failed to load. Please check the model path ('best.pt') and ensure the file is in the same directory as 'app.py'.")