import os
import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from datetime import datetime
import time
from PIL import Image
import io

# Streamlit UI Setup
st.set_page_config(page_title="Face Verification System", layout="centered")

# Keep your dark mode styling as is

st.markdown("<h1 style='text-align: center;'>Face Verification System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Capture a live image and verify it against stored images.</p>", unsafe_allow_html=True)

# Initialize session state variables
if 'verification_attempts' not in st.session_state:
    st.session_state.verification_attempts = 0
if 'verified' not in st.session_state:
    st.session_state.verified = False
if 'verification_in_progress' not in st.session_state:
    st.session_state.verification_in_progress = False

# Function to preprocess image (keeping your original function)
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

# Function to reset verification
def reset_verification():
    st.session_state.verification_attempts = 0
    st.session_state.verified = False
    st.session_state.verification_in_progress = False
    if os.path.exists("temp_live_capture.jpg"):
        os.remove("temp_live_capture.jpg")

# Function to verify face (keeping your original function)
def verify_face(image_path, applicant_folder):
    # Keeping your original verification function
    # ... (your existing verify_face function code)
    backends = ['retinaface', 'opencv']
    models = ['Facenet512']
    
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
    
    # Rest of your verification function...
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

        # Create a status placeholder
        status_placeholder = st.empty()
        
        # Use Streamlit's built-in camera input
        img_file_buffer = st.camera_input("Take a picture for verification", 
                                         key="camera",
                                         disabled=st.session_state.verification_attempts >= 3 or st.session_state.verified,
                                         help="Position your face clearly in the center of the frame.")
        
        # Process the captured image
        if img_file_buffer is not None:
            # Convert the image buffer to an OpenCV-compatible format
            bytes_data = img_file_buffer.getvalue()
            
            # Save the captured image to a file
            with open("temp_live_capture.jpg", "wb") as file:
                file.write(bytes_data)
            
            # Preprocess the image to improve face detection
            preprocess_image("temp_live_capture.jpg")
            
            # Verification process
            if st.session_state.verification_attempts < 3 and not st.session_state.verified and not st.session_state.verification_in_progress:
                st.session_state.verification_in_progress = True
                
                with st.spinner("Verifying face..."):
                    status_msg = status_placeholder.info("Comparing with stored images...")
                    
                    verification_success, message = verify_face("temp_live_capture.jpg", APPLICANT_FOLDER)
                    
                    status_placeholder.empty()
                    
                    if verification_success:
                        st.session_state.verified = True
                        status_placeholder.success(f"✅ Face Verified! Welcome, {applicant_name}.")
                        st.success("You can move to the interview next.")
                    else:
                        st.session_state.verification_attempts += 1
                        if st.session_state.verification_attempts < 3:
                            if isinstance(message, float):
                                status_placeholder.warning(f"⚠️ Face NOT Verified! You have {3 - st.session_state.verification_attempts} attempts left.")
                            else:
                                status_placeholder.warning(f"⚠️ {message} You have {3 - st.session_state.verification_attempts} attempts left.")
                            
                            # Clear the camera input to allow retaking
                            st.session_state.pop("camera")
                            st.session_state.verification_in_progress = False
                            st.experimental_rerun()
                        else:
                            status_placeholder.error("❌ Face verification failed, you cannot proceed.")
                            if st.button("Start Over with New User"):
                                reset_verification()
                                st.session_state.pop("camera", None)
                                st.experimental_rerun()
                
                st.session_state.verification_in_progress = False
            
            # For users who have exhausted their attempts
            elif st.session_state.verification_attempts >= 3:
                status_placeholder.error("❌ Face verification failed, you cannot proceed.")
                if st.button("Start Over with New User"):
                    reset_verification()
                    st.session_state.pop("camera", None)
                    st.experimental_rerun()

        # Remove the captured image if it exists and we're done
        if os.path.exists("temp_live_capture.jpg") and st.session_state.verification_attempts >= 3:
            os.remove("temp_live_capture.jpg")
