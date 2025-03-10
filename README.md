# Face Verification System

This project is an AI-powered Face Verification System built using Streamlit, OpenCV, and DeepFace. It captures a live image of the user and verifies it against the stored images of applicants. The system can be particularly useful in HR interviews or identity verification processes.

## Features
- 📸 **Live Image Capture**: Capture real-time images using your webcam.
- 🧠 **Face Verification**: Uses DeepFace for face verification against the applicant profiles.
- 💾 **Profile Management**: Stores applicant images in `APPLICANT_PROFILE` folder.
- 📜 **Resume Handling**: Includes sample resumes in `SampleResumes` folder.
- 📑 **Job Descriptions**: Contains job requirements in `job_requirements` folder.

## Installation

### Clone the Repository
```bash
git clone https://github.com/your-username/face_verification.git
cd face_verification
pip install -r requirements.txt
streamlit run app.py
```
face_verification/
│
├── APPLICANT_PROFILE/    # Stored images of applicants
├── APPLICANT_detail/     # Images captured during verification
├── SampleResumes/       # Sample applicant resumes
├── job_requirements/    # Job descriptions
├── app.py               # Main application file
├── requirements.txt     # Required Python packages
