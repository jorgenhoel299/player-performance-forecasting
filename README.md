# ⚽ Football Player Performance Prediction Using RNNs and Web Scraping 🧠

This repository hosts the code for a project that predicts the performance of professional football players using sequential data, web scraping, and machine learning. The core of the project utilizes Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, to capture time-series dependencies in football match data. The models aim to forecast Fantasy Premier League (FPL) player scores based on their past performances, team form, and opponent form. For comparison, we also implemented simpler models such as logistic and linear regression.

## 🚀 Project Overview
- **Data Source**: Web-scraped detailed player and match statistics from [fbref.com](https://fbref.com/), using Python libraries such as `BeautifulSoup` and `Selenium`. The dataset covers Premier League seasons from 2019 to 2023.
- **Models Used**:
  - **LSTM RNN**: Captures both short- and long-term player performance trends.
  - **Logistic Regression**: Predicts the likelihood of a player starting a match.
- **Feature Engineering**: Includes custom features such as player form, opponent form, and exhaustion levels.
  
## 📊 Key Features
- **Automated Web Scraping**: Efficiently scrapes thousands of CSVs with up-to-date football statistics.
- **Time-Series Forecasting**: Predicts FPL scores using sequential player and team data.
- **Performance Metrics**: Evaluates models using regression and classification metrics, including Mean Squared Error (MSE) and logistic regression for player lineup predictions.
  
## 🛠️ Tools & Technologies
- **Python**: Data scraping, preprocessing, and model development.
- **BeautifulSoup & Selenium**: Web scraping tools for gathering live football statistics. Checkout branch good-scraping and selenium-scraping for details.
- **TensorFlow/Keras**: Deep learning framework for building the LSTM model.
- **Scikit-learn**: Traditional machine learning models for comparison (Logistic Regression, Linear Regression).

## 📁 Dataset
- Thousands of CSV files representing Premier League match stats from 2019 to 2023.

