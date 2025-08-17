#  Hand Gesture Recognition & Finger Drawing 

This project combines **hand gesture recognition** and **real-time finger drawing** using [MediaPipe](https://mediapipe.dev/), OpenCV, and NumPy.  

- Recognize gestures: **Rock**, **Paper**, **Scissors**, or **None**  
- Draw with your **index finger** directly on the livestream video feed  

It’s an interactive demo that can be used for gesture-based games, digital scribbling, or live annotations.  

---

## Features

- 🎮 **Rock-Paper-Scissors Recognition**  
  Detects hand gestures and classifies them as Rock, Paper, Scissors, or None.  
  *(Depends on your trained model for classification accuracy.)*  

-  **Finger Drawing Tool**  
  Tracks your index finger in the webcam feed and lets you draw on the screen in real time.  

- 🎥 **Live Webcam Feed**  
  Processes your camera stream continuously, overlays drawings, and displays recognition results.  

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition

## 📦 Requirements

This project uses Python and a few key libraries:

- **[MediaPipe](https://developers.google.com/mediapipe)** – For real-time hand landmark detection and gesture recognition  
- **[OpenCV](https://opencv.org/)** – For video capture, drawing, and image processing  
- **[NumPy](https://numpy.org/)** – For handling image arrays and mathematical operations  
- **[protobuf](https://protobuf.dev/)** – Required by MediaPipe for model definitions  
- **[Streamlit + streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)** (optional) – For running the project in a browser-based app  

To install all dependencies, run:

```bash
pip install -r requirements.txt

