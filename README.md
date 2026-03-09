# Medicine Overdose Prediction System

## Overview
The Medicine Overdose Prediction System is a machine learning based application designed to analyze prescription information and predict the risk of potential medicine overdose. The system evaluates factors such as medicine name, dosage, and frequency to identify unsafe medication patterns and provide early warnings.

This project aims to improve patient safety and assist healthcare professionals in detecting risky prescriptions before they cause harm.

---

## Features
- Predicts potential medicine overdose risk
- Accepts manual prescription input
- Analyzes medicine name, dosage, and frequency
- Uses Machine Learning models for prediction
- Provides risk evaluation for prescriptions
- Simple and user-friendly interface

---

## Technologies Used
- Python
- Machine Learning
- Random Forest Algorithm
- Streamlit
- Pandas
- Scikit-learn
- OCR (for prescription text extraction)

---

## Project Structure
```
Medicine-Overdose-Prediction
│
├── app.py
├── dataset.csv
├── model.py
├── requirements.txt
└── README.md
```

---

## How It Works
1. User enters prescription details (medicine name, dosage, frequency).
2. The system processes the data.
3. A machine learning model analyzes the input.
4. The model predicts the risk of medicine overdose.
5. The system displays the prediction result.

---

## Installation

1. Clone the repository
```
git clone https://github.com/yourusername/Medicine-Overdose-Prediction.git
```

2. Navigate to the project folder
```
cd Medicine-Overdose-Prediction
```

3. Install required dependencies
```
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit application:

```
streamlit run app.py
```

The application will open in your browser.

---

## Future Improvements
- Integration with hospital databases
- More advanced machine learning models
- Real-time prescription analysis
- Mobile application support

---

## License
This project is licensed under the MIT License.

---

## Author
Siri Manvitha  
B.Tech Computer Science Engineering
