Sign Language Recognition System
 Project Overview

The Sign Language Recognition System is an Artificial Intelligence and Computer Vision-based application designed to recognize hand gestures in real time and convert them into readable text and speech. The project aims to bridge the communication gap between people with hearing or speech impairments and individuals who do not understand sign language.

The system uses a webcam to capture hand gestures, extracts hand landmarks using MediaPipe, and classifies gestures using a Random Forest Machine Learning model. The recognized gestures are displayed as text, and users can optionally convert the generated text into speech.

 Objectives
 
Develop a real-time sign language recognition system.
Eliminate the need for expensive hardware such as sensor gloves.
Improve communication for people with hearing and speech disabilities.
Provide a cost-effective and user-friendly assistive technology solution.
Convert sign language gestures into both text and speech.
 
 Features
 
Real-time hand gesture recognition
Webcam-based gesture detection
MediaPipe hand landmark detection
Machine Learning-based gesture classification
Text generation from recognized gestures
Text-to-Speech (TTS) conversion
Word suggestions using TextBlob
Sentence formation
Delete and Space controls
Confidence-based prediction smoothing
Streamlit web application interface

 Technologies Used
 
Programming Language
Python 3.x
Libraries
OpenCV
MediaPipe
NumPy
Pandas
Scikit-learn
Joblib
TextBlob
gTTS
Pygame
Streamlit
Matplotlib
Seaborn

 Project Structure
 
Sign-Language-Recognition-System/
│
├── dataset/
│   └── data.csv
│
├── Detection.py
├── Train_model.py
├── Prediction.py
├── app.py
│
├── sign_model.pkl
├── label_encoder.pkl


 System Requirements
 
Hardware
Intel Core i3 Processor or higher
Minimum 4 GB RAM (8 GB Recommended)
HD Webcam
Keyboard and Mouse
Software
Windows / Linux / macOS
Python 3.10 or above
Visual Studio Code / Jupyter Notebook
Streamlit

 Workflow
 
Capture hand gestures using the webcam.
Detect hand landmarks using MediaPipe.
Extract geometric features from the detected landmarks.
Train a Random Forest classifier using the collected dataset.
Predict gestures in real time.
Display the recognized gesture as text.
Convert text into speech.
Generate spelling suggestions and allow sentence formation.

 Machine Learning Model

Algorithm Used

Random Forest Classifier

Feature Extraction

Finger joint angles
Distances between landmarks
Palm orientation
Normalized hand landmark coordinates

Evaluation Metrics

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

 Installation
 
Step 1: Clone the Repository
git clone https://github.com/your-username/sign-language-recognition.git
Step 2: Navigate to the Project Folder
cd sign-language-recognition
Step 3: Install Dependencies
pip install -r requirements.txt

 How to Run
 
Collect Dataset
python Detection.py
Train the Model
python Train_model.py
Run Prediction
python Prediction.py
Launch the Web Application
streamlit run app.py

 Results
 
High gesture recognition accuracy
Real-time prediction with low latency
Reliable hand landmark detection
Accurate sentence generation
Speech synthesis support
Cost-effective implementation without specialized hardware

 Applications
 
Assistive communication
Educational institutions
Healthcare services
Public service centers
Human-computer interaction
Accessibility solutions

 Current Limitations
 
Supports only single-hand gestures
Recognizes primarily static gestures
Performance may decrease in poor lighting conditions
Limited gesture vocabulary
No multilingual translation support in the current version

 Future Enhancements
 
Dynamic gesture recognition
Two-hand gesture recognition
Mobile application support
Multilingual translation
Deep Learning models (CNN/LSTM/Transformer)
Cloud deployment
IoT integration
Personalized gesture learning
Enhanced user interface
Expanded sign language vocabulary

 Author

Ekta Kansara

MCA (Data Science)

Ajeenkya DY Patil University, Pune


 License

This project is developed for academic and educational purposes. It may be used for learning, research, and non-commercial applications with appropriate acknowledgment.
