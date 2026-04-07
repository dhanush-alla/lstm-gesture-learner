# LSTM Gesture Learner

<p align="center">
  <img width="1282" height="839" alt="image" src="https://github.com/user-attachments/assets/bfa7c8b1-9423-4d5c-928c-b705791b9df8" />
</p>

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Supported-green.svg)
![TensorFlow/Keras](https://img.shields.io/badge/Deep%20Learning-LSTM-orange.svg)

An interactive, deep learning-based application for dynamic gesture recognition. **LSTM Gesture Learner** uses Long Short-Term Memory (LSTM) neural networks to recognize continuous hand gestures. The standout feature is its ability to learn **user-taught gestures** on the fly, translating them into actions or Text-to-Speech (TTS) feedback.

## Features

* **Dynamic Gesture Recognition**: Captures spatial and temporal sequences for fluid, real-time gesture tracking.
* **Custom Gesture Learning**: Teach the model custom gestures tailored to your specific needs.
* **MediaPipe Integration**: Fast, precise, and robust hand-landmark detection.
* **LSTM Neural Networks**: Employs an LSTM-based architecture specialized for recognizing temporal sequences.
* **Text-to-Speech (TTS) Feedback**: Real-time auditory translation of recognized gestures (ideal for sign language).
* **Interactive UI**: A friendly visual interface built for seamless data collection, training, and testing.

## Project Structure

```text
lstm-gesture-learner/
├── data/               # Stores raw coordinate datasets and user-collected landmark sequences
├── models/             # Contains the saved, trained LSTM model weights (e.g., .h5 files)
├── src/                # Core scripts (data extraction, model architecture, UI logic)
├── app.py              # Main application GUI / Interface entry point
├── main.py             # Script for backend testing, CLI data collection, or training
├── requirements.txt    # Python dependencies required to run the project
├── .gitignore          # Ignored files
└── LICENSE             # MIT License
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dhanush-alla/lstm-gesture-learner.git](https://github.com/dhanush-alla/lstm-gesture-learner.git)
   cd lstm-gesture-learner
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Launching the Application

To launch the user interface and start using or teaching the model:
```bash
python app.py
```
*(Alternatively, run `python main.py` if operating via a CLI pipeline).*



### 2. Teaching a New Gesture

* Select the **Record/Add Gesture** option in the app.
* Enter a label for your new gesture.
* Perform the gesture smoothly in front of the camera. The system will record the sequential landmarks.
* Trigger the **Train** function to update the LSTM weights with the newly collected data.

<p align="center">
  <img width="1282" height="839" alt="image" src="https://github.com/user-attachments/assets/d48cb7f6-aa19-4670-b576-3ca8fced61a2" />

</p>

### 3. Gesture Recognition

* Switch to **Prediction/Recognition** mode.
* Perform a trained gesture. The application will track your hand, predict the sequence using the LSTM model, and output the result visually and audibly via Text-to-Speech.

## How It Works

<p align="center">
  <img width="1131" height="605" alt="image" src="https://github.com/user-attachments/assets/636e0f5b-0ca9-426a-b0e0-14bb4b719630" />
</p>

1. **Feature Extraction:** OpenCV accesses the webcam, and MediaPipe isolates the 21 3D landmarks of the human hand frame-by-frame.
2. **Sequence Padding & Formatting:** The extracted landmarks are flattened and stored as a sequence of frames representing the motion over time.
3. **LSTM Inference:** The time-series data is fed into the LSTM Deep Learning model, which is highly capable of connecting the context between the previous and current frames to predict the final action.

## Core Dependencies

* `OpenCV` (Computer Vision)
* `MediaPipe` (Hand tracking/landmarks)
* `TensorFlow` / `Keras` (LSTM model building and training)
* `pyttsx3` / `gTTS` (Text-to-speech synthesis)
* `NumPy` & `Pandas` (Data manipulation)

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues). 

---
*Built by [Dhanush Alla](https://github.com/dhanush-alla)*
