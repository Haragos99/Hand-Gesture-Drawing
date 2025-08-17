# 🖐️ Hand Gesture Recognition & Finger Drawing  

This project combines **hand gesture recognition** with a **real-time finger drawing tool** built using [MediaPipe](https://mediapipe.dev/), OpenCV, and NumPy.  

With this demo, you can:  
- Recognize hand gestures (**Rock**, **Paper**, **Scissors**, or **None**)  
- Draw on the **live camera feed** using just your index finger  

It’s designed as an interactive prototype for gesture-based games, digital whiteboarding, or live annotations.  

---

## ✨ Features

- 🎮 **Rock-Paper-Scissors Recognition**  
  Detects hand gestures and classifies them as **Rock**, **Paper**, **Scissors**, or **None**.  
  *(Accuracy depends on your trained recognition model.)*  

- ✏️ **Finger Drawing on Live Feed**  
  Tracks the index finger in real time, enabling drawing directly on the webcam stream.  

- 🎥 **Continuous Live Video**  
  Uses your webcam to process frames, overlay gesture recognition, and display finger drawings instantly.  

---

## ⚙️ Requirements & Installation  

### Prerequisites  
- Python **3.8+**  
- A working webcam  

### Dependencies  
This project relies on the following Python libraries:  
- [MediaPipe](https://pypi.org/project/mediapipe/) – Hand tracking & gesture recognition  
- [OpenCV](https://pypi.org/project/opencv-python/) – Video capture & image processing  
- [NumPy](https://pypi.org/project/numpy/) – Numerical operations  

### Installation  

Clone the repository and install the requirements:  

```
git clone https://github.com/your-username/hand-gesture-drawing.git
cd hand-gesture-drawing
pip install -r requirements.txt
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition
```
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
```
