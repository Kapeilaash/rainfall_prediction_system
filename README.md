# 🌧️ Rainfall Prediction System Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive machine learning system that predicts rainfall occurrence using meteorological data. This project implements advanced data preprocessing techniques, exploratory data analysis, and optimized Random Forest classification to achieve accurate weather predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

The Rainfall Prediction System leverages machine learning algorithms to predict rainfall based on various meteorological parameters. This project addresses the critical need for accurate weather forecasting in agriculture, disaster management, and daily planning.

### Key Objectives:
- **Accurate Prediction**: Achieve high accuracy in binary rainfall classification
- **Data-Driven Insights**: Extract meaningful patterns from weather data
- **Scalable Solution**: Build a robust system that can handle new weather data
- **Operational Efficiency**: Provide quick predictions for real-time decision making

## 📊 Dataset

The system uses a comprehensive rainfall dataset with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| Pressure | Atmospheric pressure (hPa) | Numerical |
| Temperature | Current temperature (°C) | Numerical |
| Max Temperature | Maximum daily temperature (°C) | Numerical |
| Min Temperature | Minimum daily temperature (°C) | Numerical |
| Dewpoint | Dew point temperature (°C) | Numerical |
| Humidity | Relative humidity (%) | Numerical |
| Cloud | Cloud cover percentage (%) | Numerical |
| Sunshine | Hours of sunshine | Numerical |
| Wind Direction | Wind direction (degrees) | Numerical |
| Wind Speed | Wind speed (km/h) | Numerical |
| Rainfall | Target variable (Yes/No) | Categorical |

**Dataset Statistics:**
- Total samples: 366 records
- Features: 10 meteorological parameters
- Target classes: Binary (Rainfall/No Rainfall)
- Missing values: Handled through advanced imputation

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kapeilaash/rainfall_prediction_system.git
   cd rainfall_prediction_system
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv rainfall_env
   source rainfall_env/bin/activate  # On Windows: rainfall_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

### Required Libraries
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
pickle5>=0.0.11
```

## 💻 Usage

### Quick Start

1. **Open the main notebook:**
   ```bash
   jupyter notebook Rainfall_Prediction_using_Machine_Learning.ipynb
   ```

2. **Run all cells sequentially** to:
   - Load and explore the dataset
   - Perform data preprocessing
   - Train the machine learning model
   - Evaluate model performance
   - Make predictions on new data

### Making Predictions

```python
import pickle
import pandas as pd

# Load the trained model
with open('rainfall_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example prediction
new_data = {
    'pressure': 1015.9,
    'dewpoint': 19.9,
    'humidity': 95,
    'cloud': 81,
    'sunshine': 0.0,
    'winddirection': 40.0,
    'windspeed': 13.7
}

# Create DataFrame
input_df = pd.DataFrame([new_data])

# Make prediction
prediction = model.predict(input_df)
result = "Rainfall Expected" if prediction[0] == 1 else "No Rainfall"
print(f"Prediction: {result}")
```

## 📁 Project Structure

```
rainfall_prediction_system/
├── 📊 raining dataset/
│   └── Rainfall.csv                    # Raw meteorological data
├── 📓 Rainfall_Prediction_using_Machine_Learning.ipynb  # Main analysis notebook
├── 🤖 rainfall_prediction_model.pkl    # Trained model (generated)
├── 📋 requirements.txt                 # Python dependencies
├── 📖 README.md                       # Project documentation
                  # MIT license file
```

## 🔬 Methodology

### 1. Data Preprocessing Pipeline
- **Missing Value Treatment**: Strategic imputation for wind direction and speed
- **Feature Engineering**: Categorical encoding and normalization
- **Data Cleaning**: Outlier detection and handling
- **Feature Selection**: Correlation-based redundant feature removal

### 2. Exploratory Data Analysis
- **Statistical Analysis**: Descriptive statistics and distribution analysis
- **Visualization**: Comprehensive plots using matplotlib and seaborn
- **Correlation Analysis**: Feature relationship identification
- **Class Balance Assessment**: Target variable distribution analysis

### 3. Model Development
- **Algorithm Selection**: Random Forest Classifier
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
- **Class Balancing**: Down-sampling technique for balanced dataset
- **Model Validation**: Robust cross-validation strategy

### 4. Performance Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Validation**: K-fold cross-validation
- **Error Analysis**: Confusion matrix and classification report

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~72% |
| **Precision** | High precision for both classes |
| **Recall** | Balanced recall across classes |
| **F1-Score** | Optimized F1 score |
| **Cross-validation** | Consistent performance across folds |

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Best Parameters**: Optimized through GridSearchCV
- **Training Data**: 234 balanced samples
- **Validation Method**: 5-fold cross-validation

## ✨ Features

### Core Functionality
- 🔍 **Automated Data Preprocessing**: Handles missing values and data cleaning
- 📊 **Comprehensive EDA**: Statistical analysis and visualization
- 🎯 **Feature Selection**: Correlation-based feature optimization
- ⚖️ **Class Balancing**: Addresses dataset imbalance issues
- 🔧 **Hyperparameter Tuning**: GridSearchCV optimization
- 💾 **Model Persistence**: Pickle-based model saving/loading
- 🎲 **Real-time Prediction**: Easy prediction interface

### Technical Highlights
- **Scalable Architecture**: Modular code design
- **Robust Validation**: Multiple validation techniques
- **Performance Monitoring**: Comprehensive evaluation metrics
- **Documentation**: Well-documented codebase

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

### Areas for Contribution
- [ ] Additional machine learning algorithms
- [ ] Real-time weather API integration
- [ ] Web application development
- [ ] Mobile app implementation
- [ ] Performance optimization
- [ ] Documentation improvements

## 🔮 Future Roadmap

- [ ] **Web Deployment**: Flask/Django web application
- [ ] **API Integration**: Real-time weather data feeds
- [ ] **Mobile App**: Cross-platform mobile application
- [ ] **Advanced Models**: Deep learning implementations
- [ ] **Regional Analysis**: Location-specific predictions
- [ ] **Time Series**: Long-term forecasting capabilities

## 👨‍💻 Contact

**Kapeilaash**
- 🐙 GitHub: [@Kapeilaash](https://github.com/Kapeilaash)
- 📧 Email: [koneswarankapeilaash@gmail.com]
- 💼 LinkedIn: [https://www.linkedin.com/in/kapeilaash-k/]

## 🙏 Acknowledgments

- Weather data providers and meteorological organizations
- Scikit-learn development team
- Open-source machine learning community
- Contributors and users of this project

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

[🔝 Back to Top](#-rainfall-prediction-system-using-machine-learning)

</div>

