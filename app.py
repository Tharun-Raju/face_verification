import os
import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from datetime import datetime
import time
import io
from PIL import Image, ImageEnhance

# Streamlit UI Setup
st.set_page_config(page_title="Face Verification System", layout="centered")

# Dark Mode Styling
st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input {
            background-color: #1e1e1e;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #125f88;
        }
        .stImage {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center;'>Face Verification System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Capture a live image and verify it against stored images.</p>", unsafe_allow_html=True)

# Initialize session state variables
if 'image_captured' not in st.session_state:
    st.session_state.image_captured = False
if 'verification_attempts' not in st.session_state:
    st.session_state.verification_attempts = 0
if 'verified' not in st.session_state:
    st.session_state.verified = False
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None
if 'verification_in_progress' not in st.session_state:
    st.session_state.verification_in_progress = False

# Enhanced image preprocessing function
def enhance_image(img):
    """Apply multiple image enhancement techniques to improve face recognition."""
    try:
        # Convert to PIL Image for enhancement
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = img
            
        # 1. Enhance brightness
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(1.2)  # Slightly increase brightness
        
        # 2. Enhance contrast
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(1.3)  # Increase contrast
        
        # 3. Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(1.5)  # Increase sharpness
        
        # Convert back to OpenCV format
        img_enhanced = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 4. Apply additional OpenCV enhancements
        # Resize to ensure proper dimensions
        img_enhanced = cv2.resize(img_enhanced, (640, 480))
        
        # Apply CLAHE for adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        img_enhanced = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
        
        return img_enhanced
    
    except Exception as e:
        st.error(f"Error enhancing image: {e}")
        # Return original image if enhancement fails
        if isinstance(img, np.ndarray):
            return img
        return np.array(img)

# Modified function to preprocess and save image
def process_and_save_image(img_data, save_path):
    try:
        # If img_data is bytes, convert to cv2 image
        if isinstance(img_data, bytes):
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = img_data
            
        # Apply image enhancements
        enhanced_img = enhance_image(img)
        
        # Save the enhanced image
        cv2.imwrite(save_path, enhanced_img)
        return enhanced_img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to reset verification
def reset_verification():
    st.session_state.verification_attempts = 0
    st.session_state.image_captured = False
    st.session_state.verified = False
    st.session_state.captured_frame = None
    st.session_state.verification_in_progress = False
    if os.path.exists("temp_live_capture.jpg"):
        os.remove("temp_live_capture.jpg")

# Improved face verification function with more robust parameters
def verify_face(image_path, applicant_folder):
    # Create directory for pre-processed faces if it doesn't exist
    processed_dir = "temp_processed_faces"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Process the captured image for different models
    try:
        # Create multiple processed versions with different parameters
        img = cv2.imread(image_path)
        processed_paths = []
        
        # Standard image
        processed_path = f"{processed_dir}/standard.jpg"
        cv2.imwrite(processed_path, img)
        processed_paths.append(processed_path)
        
        # Slightly different brightness/contrast versions
        for i, brightness in enumerate([0.9, 1.1]):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(brightness)
            processed_path = f"{processed_dir}/brightness_{i}.jpg"
            img_pil.save(processed_path)
            processed_paths.append(processed_path)
        
        # Use all backends for better detection chances
        backends = ['retinaface', 'opencv', 'mtcnn', 'ssd', 'dlib']
        models = ['Facenet512', 'ArcFace', 'VGG-Face']
        
        # First check if a face is detectable
        face_detected = False
        for backend in backends:
            try:
                DeepFace.detectFace(
                    img_path=image_path, 
                    target_size=(224, 224), 
                    detector_backend=backend
                )
                face_detected = True
                break
            except Exception:
                continue
        
        if not face_detected:
            return False, "No face detected. Please try again with better lighting."
        
        # Proceed with verification if face was detected
        best_distance = 1.0
        verification_success = False
        
        # Check against stored images with timeout to prevent hanging
        start_time = time.time()
        timeout = 10  # Increased timeout for more thorough checking
        
        for file in os.listdir(applicant_folder):
            # Check if we've spent too much time already
            if time.time() - start_time > timeout:
                break
                
            if file.endswith(('.jpg', '.jpeg', '.png')):
                stored_image_path = os.path.join(applicant_folder, file)
                
                # Try verification with each model/backend combination until success
                for model in models:
                    for backend in backends[:2]:  # Limit to first 2 backends for speed
                        for processed_path in processed_paths:
                            try:
                                result = DeepFace.verify(
                                    img1_path=processed_path,
                                    img2_path=stored_image_path,
                                    model_name=model,
                                    distance_metric='cosine',
                                    detector_backend=backend
                                )
                                
                                if result["verified"] and result["distance"] < best_distance:
                                    best_distance = result["distance"]
                                
                                # More lenient threshold to improve recognition
                                if result["verified"] or best_distance < 0.65:  # Increased threshold
                                    verification_success = True
                                    break
                            except Exception:
                                continue
                        
                        if verification_success:
                            break
                    if verification_success:
                        break
                if verification_success:
                    break
        
        # Clean up processed images
        for path in processed_paths:
            if os.path.exists(path):
                os.remove(path)
        
        if os.path.exists(processed_dir):
            os.rmdir(processed_dir)
        
        return verification_success, best_distance if verification_success else None
    
    except Exception as e:
        st.error(f"Error during verification: {e}")
        return False, str(e)

# Main app flow
applicant_name = st.text_input("Enter applicant name:", placeholder="Type name and press Enter...").strip()

if applicant_name:
    APPLICANT_FOLDER = f"APPLICANT_PROFILE/{applicant_name}"
    
    if not os.path.exists(APPLICANT_FOLDER):
        st.error(f"No folder found for {applicant_name}! Please add images.")
    else:
        st.success(f"Folder found: {APPLICANT_FOLDER}")

        # Camera placeholder for displaying the captured image
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Add option to upload image as alternative to camera
        st.markdown("### Choose how to verify your face:")
        option = st.radio("", ["Use Camera", "Upload Image"], horizontal=True)
        
        if option == "Use Camera":
            # Display instructions
            if not st.session_state.image_captured:
                st.info("Position your face clearly in the center of the frame and take a photo in good lighting.")
                st.warning("For best results, ensure your face is well-lit and directly facing the camera.")
                
                # Offer camera quality options
                camera_quality = st.select_slider(
                    "Camera Quality (higher is better but slower)",
                    options=["Low", "Medium", "High"],
                    value="High"
                )
                
                # Map quality settings to actual parameters
                quality_map = {
                    "Low": 0.5,
                    "Medium": 0.75,
                    "High": 1.0
                }
                
                # Use Streamlit's native camera input
                camera_image = st.camera_input("Take a photo", key=f"camera_{quality_map[camera_quality]}")
                
                if camera_image:
                    # Process the captured image
                    bytes_data = camera_image.getvalue()
                    
                    # Process and save the enhanced image
                    enhanced_img = process_and_save_image(bytes_data, "temp_live_capture.jpg")
                    
                    if enhanced_img is not None:
                        # Update session state
                        st.session_state.captured_frame = enhanced_img
                        st.session_state.image_captured = True
                        
                        # Force rerun to update the UI
                        st.rerun()
        
        else:  # Upload Image option
            if not st.session_state.image_captured:
                st.info("Upload a clear photo of your face.")
                
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
                
                if uploaded_file is not None:
                    # Read and process the uploaded image
                    image = Image.open(uploaded_file)
                    
                    # Convert PIL image to OpenCV format and enhance
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    enhanced_img = process_and_save_image(img_cv, "temp_live_capture.jpg")
                    
                    if enhanced_img is not None:
                        # Update session state
                        st.session_state.captured_frame = enhanced_img
                        st.session_state.image_captured = True
                        
                        # Force rerun to update the UI
                        st.rerun()
        
        # Display captured image
        if st.session_state.image_captured and os.path.exists("temp_live_capture.jpg"):
            captured_img = cv2.imread("temp_live_capture.jpg")
            if captured_img is not None:
                captured_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(captured_rgb, caption="Captured Image", use_container_width=True)
            
            # Verification process
            if st.session_state.verification_attempts < 3 and not st.session_state.verified and not st.session_state.verification_in_progress:
                image_path = "temp_live_capture.jpg"
                st.session_state.verification_in_progress = True
                
                # Use the global spinner with a maximum wait time
                with st.spinner("Verifying face..."):
                    # Create a background message for longer verifications
                    status_msg = status_placeholder.info("Comparing with stored images... This may take a moment.")
                    
                    verification_success, message = verify_face(image_path, APPLICANT_FOLDER)
                    
                    # Clear the status message
                    status_placeholder.empty()
                    
                    if verification_success:
                        st.session_state.verified = True
                        status_placeholder.success(f"✅ Face Verified! Welcome, {applicant_name}.")
                        # Changed this section to display the interview prompt instead of "New Verification" button
                        st.success("You can move to the interview next.")
                    else:
                        st.session_state.verification_attempts += 1
                        if st.session_state.verification_attempts < 3:
                            if isinstance(message, float):
                                status_placeholder.warning(f"⚠️ Face NOT Verified! You have {3 - st.session_state.verification_attempts} attempts left.")
                            else:
                                status_placeholder.warning(f"⚠️ {message} You have {3 - st.session_state.verification_attempts} attempts left.")
                            
                            if st.button("Try Again"):
                                st.session_state.image_captured = False
                                st.session_state.verification_in_progress = False
                                st.rerun()
                        else:
                            # Changed this section to display the failure message
                            status_placeholder.error("❌ Face verification failed, you cannot proceed.")
                            if st.button("Start Over with New User"):
                                reset_verification()
                                st.experimental_set_query_params()
                                st.rerun()
                
                st.session_state.verification_in_progress = False
            elif st.session_state.verification_attempts >= 3:
                # This section is also updated with the new failure message
                status_placeholder.error("❌ Face verification failed, you cannot proceed.")
                if st.button("Start Over with New User"):
                    reset_verification()
                    st.experimental_set_query_params()
                    st.rerun()

        # Remove the captured image if it exists
        if os.path.exists("temp_live_capture.jpg") and st.session_state.verification_attempts >= 3:
            os.remove("temp_live_capture.jpg")
