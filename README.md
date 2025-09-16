# AI Project - Plastic Bottle Label Detection

A deep learning project that detects plastic bottles and classifies them as **with label** or **without label**. The system integrates a trained CNN classifier with a YOLO object detector and connects to an ESP32 microcontroller to control LEDs:
- 🟢 Green LED → Bottle with label
- 🔴 Red LED → Bottle without label

## Team Members

| AC.NO     | Name | Role | Contributions |
|----|------|------|---------------|
| 202274275 | Jabr Saleh Ali | Lead Developer | Data preprocessing, model development, ESP32 integration |
| 202274219 | Mubarak Al-salami| Data Analyst | Dataset collection, visualization, EDA |
| 202073141 | Ayham alkabodi | ML Engineer | Model training, optimization, evaluation |
| 202274088 | Zakria Hajam | ML Engineer | Model training, testing, evaluation |

---

## Installation and Setup

### Prerequisites
- Python 3.9.13+
- UV package manager (for dependency management)
- Webcam or DroidCam (for camera input)
- ESP32 microcontroller (for LED output)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/mubarak-mub/Plastic-bottle-detection.git
   cd Plastic-bottle-detection
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Run the detection project:
   ```bash
   uv run python main.py
   ```

4. Run model evaluation on test images:
   ```bash
   uv run python train.py
   ```

---

## Project Structure

```
bottle-label-detection/
├── README.md              # Project documentation
├── pyproject.toml         # UV project configuration
├── .python-version        # Python version specification
├── main.py                # Real-time detection with YOLO + CNN
├── train.py               # Evaluate model on test dataset
├── esp32/                 # ESP32 microcontroller integration code
├── data/                  # Dataset (train/val/test images)
├── notebooks/             # Jupyter notebooks for experiments
├── Src/                   # contain module data and utlis
└── docs/                  # Documentation and results
```

---

## Usage

### Real-time Detection
```bash
uv run python camera_detection.py
```
- Detects bottles in live video stream using YOLO.  
- Classifies each bottle ROI as **with_label** or **without_label** using CNN.  
- Shows results on the screen and sends output to ESP32.  

### Model Evaluation
```bash
uv run python test_evaluation.py
```
- Runs evaluation on ~200+ test images.  
- Reports Accuracy, Precision, Recall, and F1-Score.  

---



## Results

- **Accuracy**: 98.67%
- **Precision (with_label)**: 97.78%
- **Recall (with_label)**: 99.77%
- **F1-Score (with_label)**: 98.77%
- **Precision (without_label)**: 99.74%
- **Recall (without_label)**: 97.42%
- **F1-Score (without_label)**: 98.57%

**Key Findings**:  
- CNN + YOLO pipeline is effective for distinguishing labeled vs. unlabeled bottles.  
- Dataset diversity improves robustness.  
- Integration with ESP32 enables real-time hardware feedback.  

---

## Hardware Integration (ESP32)

- 🟢 Green LED lights up if bottle has a label.  
- 🔴 Red LED lights up if bottle has no label.  
- Communication via serial between PC and ESP32.  

---

## Contributing

1. Fork the repository  
2. Create a feature branch:  
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes  
4. Commit changes:  
   ```bash
   git commit -m 'Add feature'
   ```
5. Push to branch:  
   ```bash
   git push origin feature-name
   ```
6. Submit a pull request  
