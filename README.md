

# 🚗 Driver Drowsiness Detection using Deep Learning

This project detects driver drowsiness in real-time using a **hybrid deep learning model** that combines **VGG16** and **DenseNet121**. The system analyzes eye states (open/closed) from images or video streams and triggers an alert via **ThingSpeak IoT** when drowsiness is detected.



## 📌 Features

* Uses **VGG16 + DenseNet121** for robust feature extraction.
* **Transfer Learning** with fine-tuning on a custom dataset of eye images.
* **Real-time video analysis** using OpenCV.
* **IoT integration** with **ThingSpeak API** to send alerts.
* **Data preprocessing pipeline** for splitting into training & testing sets.
* **Visualization** of accuracy and loss during training.



## 📂 Dataset

The dataset used is from Kaggle:
[Closed and Open Eyes Dataset](https://www.kaggle.com) 

Structure after preprocessing:

```
driver_drowsiness_dataset_split/
│── train/
│   ├── drowsy/
│   ├── notdrowsy/
│── test/
    ├── drowsy/
    ├── notdrowsy/
```

---

## ⚙️ Installation & Setup

### 🔹 1. Clone the repository

```bash
git clone https://github.com/Priyankagarhwal/drowsiness_detection.git
cd drowsiness_detection
```

### 🔹 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 🔹 3. (Optional) Running in Google Colab

If running in **Colab**:

* Enable **GPU runtime**
* Upload your **Kaggle API key (`kaggle.json`)**
* Run:

  ```python
  from google.colab import files
  files.upload()  # upload kaggle.json
  ```
* Download dataset with Kaggle API
* Update dataset paths to `/content/...`

---

## 🧠 Model Architecture

* Input: `224x224x3` images
* Feature extractors:

  * **VGG16** (pretrained on ImageNet, fine-tuned last 4 layers)
  * **DenseNet121** (pretrained on ImageNet, fine-tuned last 4 layers)
* Combined using **Concatenation + Fully Connected Layers**
* Final Layer: **Sigmoid (binary classification)**

---

## 📊 Training

```python
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
```

* Best model saved as:
  `vgg16_densenet_drowsiness_best.keras`
* Final model saved as:
  `vgg16_densenet_drowsiness_final.keras`

---

## 🎥 Running on Video

1. Place your video in the project directory.
2. Update path in the script:

   ```python
   v = cv2.VideoCapture("your_video.mp4")
   ```
3. Run the script:

   ```bash
   python detect_drowsiness.py
   ```

---

## 🌐 IoT Integration with ThingSpeak

* Uses **ThingSpeak API** to post drowsiness alerts.
* Example payload:

  ```python
  KEY = "YOUR_API_KEY"
  URL = "https://api.thingspeak.com/update"
  payload = {"api_key": KEY, "field1": 1}  # 1 = Drowsy, 0 = Awake
  ```

---

## 📈 Results

* Achieved high accuracy in detecting drowsiness.
* Real-time detection tested on video footage.
* Alerts successfully triggered on ThingSpeak dashboard.

---

## 🚀 Future Work

* Extend to real-time webcam monitoring.
* Deploy as a mobile app.
* Integrate with vehicle alert systems (buzzer/vibration).

---

## 🙌 Acknowledgements

* Kaggle dataset providers
* TensorFlow & Keras libraries
* ThingSpeak IoT platform

