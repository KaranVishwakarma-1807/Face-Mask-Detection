# Face Mask Detection with TensorFlow & OpenCV
---

# Objective
Detect if a person is wearing a face mask in real-time using computer vision and deep learning.
---

# Features
- Image classification (mask vs no mask)
- Real-time face detection and mask prediction (OpenCV)
- Model training and evaluation
---

# Project Setup
1. **Clone the repository**
- git clone https://github.com/yourusername/face-mask-detection.git
- cd face-mask-detection

2. **Install dependencies**
- pip install -r requirements.txt

3. **Prepare dataset**  
- Use this dataset (Kaggle): [Face Mask Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-detection)  
- Structure it like this:
<br/>
dataset/
<br/>
├── train/
<br/>
│ ├── with_mask/
<br/>
│ └── without_mask/
<br/>
└── val/
<br/>
├── with_mask/
<br/>
└── without_mask/
<br/>
<br/>

4. **Train the model**
- python mask_detector.py

5. **Run real-time mask detection**
- python detect_mask_video.py
---

# Real-World Application
Public safety checks in areas requiring mask-wearing, such as:
- Airports
- Hospitals
- Offices
