# Insurance Cost Prediction Web App

This Streamlit web application predicts the medical insurance cost based on user-provided information and provides recommendations to reduce the cost. The app also includes interactive data visualizations that provide insights into how various factors affect insurance costs.

## Features

- **Insurance Cost Prediction**: Predicts medical insurance cost based on age, sex, BMI, smoking status, number of children, and region.
- **Personalized Recommendations**: Provides suggestions to reduce insurance costs, including tips for BMI, smoking habits, and family plans.
- **Data Visualizations**:
  - Distribution of insurance charges.
  - Impact of BMI on insurance costs with smoker status filter.
  - Age vs. charges with age range selection.
  - Average charges by region.

## Technology Stack

- **Frontend**: Streamlit for interactive UI
- **Machine Learning**: Gradient Boosting Regressor for prediction model
- **Data Visualization**: Plotly for interactive charts
- **Model Persistence**: Joblib for saving and loading the trained model

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Chiranjeevisai18/insurance-cost-prediction.git
   cd insurance-cost-prediction
