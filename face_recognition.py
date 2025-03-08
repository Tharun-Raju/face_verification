import os
import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from datetime import datetime
import time

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
if 'preview_active' not in st.session_state:
    st.session_state.preview_active = False
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

# Function to toggle camera state
def toggle_camera():
    st.session_state.preview_active = not st.session_state.preview_active
    st.session_state.image_captured = False

# Function to preprocess image to improve face detection
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        # Resize to reasonable dimensions
        img = cv2.resize(img, (640, 480))
        
        # Improve contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Save the preprocessed image
        cv2.imwrite(image_path, img)
        return True
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return False

# Function to capture image with improved reliability
def capture_image():
    cap = cv2.VideoCapture(0)
    
    # Allow camera to warm up/stabilize
    for _ in range(3):  # Reduced from 5 to 3 frames
        cap.read()
        time.sleep(0.05)  # Reduced from 0.1 to 0.05
    
    # Take just one good frame instead of multiple
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        st.session_state.captured_frame = frame
        st.session_state.image_captured = True
        st.session_state.preview_active = False
        
        # Save the captured image
        cv2.imwrite("temp_live_capture.jpg", frame)
        
        # Preprocess the image to improve face detection
        preprocess_image("temp_live_capture.jpg")
    else:
        st.error("Failed to capture image from camera!")

# Function to reset verification
def reset_verification():
    st.session_state.verification_attempts = 0
    st.session_state.image_captured = False
    st.session_state.verified = False
    st.session_state.captured_frame = None
    st.session_state.verification_in_progress = False
    if os.path.exists("temp_live_capture.jpg"):
        os.remove("temp_live_capture.jpg")

# Function to verify face with optimized performance
def verify_face(image_path, applicant_folder):
    # Use only the most reliable backends and models to improve speed
    backends = ['retinaface', 'opencv']  # Limited to just 2 backends
    models = ['Facenet512']  # Using just the best model for speed
    
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
    timeout = 5  # Maximum seconds to spend on verification
    
    for file in os.listdir(applicant_folder):
        # Check if we've spent too much time already
        if time.time() - start_time > timeout:
            break
            
        if file.endswith(('.jpg', '.jpeg', '.png')):
            stored_image_path = os.path.join(applicant_folder, file)
            
            try:
                result = DeepFace.verify(
                    img1_path=image_path,
                    img2_path=stored_image_path,
                    model_name=models[0],
                    distance_metric='cosine',
                    detector_backend=backends[0]
                )
                
                if result["verified"] and result["distance"] < best_distance:
                    best_distance = result["distance"]
                    
                if best_distance < 0.55:  # Back to original threshold
                    verification_success = True
                    break
                    
            except Exception:
                continue
    
    return verification_success, best_distance if verification_success else None

# Main app flow
applicant_name = st.text_input("Enter applicant name:", placeholder="Type name and press Enter...").strip()

if applicant_name:
    APPLICANT_FOLDER = f"APPLICANT_PROFILE/{applicant_name}"
    
    if not os.path.exists(APPLICANT_FOLDER):
        st.error(f"No folder found for {applicant_name}! Please add images.")
    else:
        st.success(f"Folder found: {APPLICANT_FOLDER}")

        # Camera placeholder for live feed
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Create camera control buttons
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.preview_active and not st.session_state.image_captured:
                start_camera = st.button("Start Camera", on_click=toggle_camera)
            elif st.session_state.preview_active:
                stop_camera = st.button("Stop Camera", on_click=toggle_camera)
        
        with col2:
            if st.session_state.preview_active:
                capture_btn = st.button("Capture Image", on_click=capture_image)
        
        # Handle live camera feed
        if st.session_state.preview_active:
            cap = cv2.VideoCapture(0)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                st.error("Error: Could not open camera.")
                st.session_state.preview_active = False
            else:
                # Add camera instructions
                st.info("Position your face clearly in the center of the frame.")
                
                # Create a frame to display the live feed
                while st.session_state.preview_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to get frame from camera.")
                        break
                    
                    # Convert color from BGR to RGB (what Streamlit expects)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Draw face detection guide overlay
                    height, width = frame.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    radius = min(width, height) // 4
                    cv2.circle(frame_rgb, (center_x, center_y), radius, (0, 255, 0), 2)
                    
                    camera_placeholder.image(frame_rgb, caption="Live Camera Feed", use_container_width=True)
                    
                    # Add a small delay to not overload the UI
                    time.sleep(0.1)
                    
                    # Check if any interactions happened
                    if not st.session_state.preview_active:
                        break
                
                cap.release()
        
        # Display captured image
        if st.session_state.image_captured and st.session_state.captured_frame is not None:
            captured_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(captured_rgb, caption="Captured Image", use_container_width=True)
            
            # Verification process
            if st.session_state.verification_attempts < 3 and not st.session_state.verified and not st.session_state.verification_in_progress:
                image_path = "temp_live_capture.jpg"
                st.session_state.verification_in_progress = True
                
                # Use the global spinner with a maximum wait time
                with st.spinner("Verifying face..."):
                    # Create a background message for longer verifications
                    status_msg = status_placeholder.info("Comparing with stored images...")
                    
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
                            
                            if st.button("Retake Image"):
                                st.session_state.image_captured = False
                                st.session_state.preview_active = True
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