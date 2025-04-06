
# 🏘️ Advanced Real Estate Analytics & Prediction Platform

## 📑 Overview

The **Advanced Real Estate Analytics & Prediction Platform** is a powerful, data-driven solution designed to transform real estate analysis and forecasting. Utilizing historical transaction data from 2001 to 2022, this platform combines cutting-edge machine learning, geospatial visualization, and an intuitive Streamlit interface. It equips real estate professionals, investors, and analysts with precise price predictions, market insights, and investment opportunity identification.

---

## 🌟 Key Features

- **🔍 Price Prediction**: Ensemble models provide accurate property price forecasts with uncertainty quantification.
- **📊 Market Segmentation**: Clustering techniques categorize properties into actionable segments.
- **📈 Feature Importance**: SHAP values highlight key drivers of property valuations.
- **🗺️ Geospatial Visualization**: Interactive maps display price distributions and value opportunities.
- **⏳ Time Series Forecasting**: 12-month market trend predictions with confidence intervals.
- **💰 Investment Opportunities**: Custom tool to pinpoint high-ROI properties based on user criteria.

---

## 🚀 Installation

Follow these steps to set up and run the platform locally:

### Prerequisites
- Python 3.8 or higher
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/mansityagi01/Real_Estate_Price_predictor.git
   cd Real_Estate_Price_predictor
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Install the required libraries with these commands:
   ```bash
   pip install streamlit
   pip install pandas
   pip install numpy
   pip install matplotlib
   pip install seaborn
   pip install plotly
   pip install folium
   pip install streamlit-folium
   pip install scikit-learn
   pip install xgboost
   pip install lightgbm
   pip install shap
   pip install pillow
   ```

4. **Obtain the Dataset**
   - The platform uses the Real Estate Sales 2001-2018 dataset from:  
     [https://catalog.data.gov/dataset/real-estate-sales-2001-2018](https://catalog.data.gov/dataset/real-estate-sales-2001-2018)
   - Download the dataset and save it as `Raw data for dashboard.csv` in the project root, or update the file path in the script.

5. **Launch the Application**
   - Ensure your script is named `main.py`.
   ```bash
   streamlit run main.py
   ```
   - Open your browser to `http://localhost:8501` to access the platform.

---

## 🛠️ Technical Architecture

The platform is built on a robust foundation:

- **Data Preprocessing**: Robust scaling, KNN imputation, and anomaly detection ensure data integrity.
- **Feature Engineering**: Temporal, geospatial, and market-derived features enhance predictions.
- **Machine Learning**: Ensemble of Gradient Boosting, XGBoost, LightGBM, Random Forest, and ElasticNet.
- **Optimization**: Bayesian hyperparameter tuning for optimal model performance.
- **Visualization**: Plotly for interactive charts, Folium for maps, and Seaborn for statistical insights.
- **Interface**: Streamlit delivers a seamless, user-friendly experience.

---

## 📊 Dataset

The platform leverages the **Real Estate Sales 2001-2018 dataset**, available at:  
[https://catalog.data.gov/dataset/real-estate-sales-2001-2018](https://catalog.data.gov/dataset/real-estate-sales-2001-2018).  

Key variables include:
- Sale Amount
- Assessed Value
- Property Type
- Town
- Date Recorded

Additional engineered features like Price-to-Assessment Ratio and Price per Square Foot enrich the analysis.

---

## 🖥️ Usage

Once launched, users can:
- **Filter Data**: Use the sidebar to refine by location, property type, price range, and time period.
- **Explore Tabs**:
  1. **🏠 Price Prediction**: Generate forecasts with confidence intervals and comparable sales.
  2. **📊 Market Insights**: Analyze trends, forecasts, and segment profiles.
  3. **📈 Performance Analysis**: Review model metrics and error distributions.
  4. **🔍 Feature Importance**: Understand valuation drivers.
  5. **🗺️ Geospatial Analysis**: Visualize price and value distributions.
  6. **💰 Investment Opportunities**: Identify properties matching investment goals.
- **Export Insights**: Extract results for further use.

---

## 🎯 Target Audience

- **Real Estate Investors**: Identify undervalued properties and estimate returns.
- **Realtors**: Provide clients with data-driven pricing and market insights.
- **Market Analysts**: Study trends and segment dynamics.
- **Homebuyers**: Make informed purchasing decisions.

---

## 🤝 Contributing

Contributions are welcome to improve the platform. To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/new-functionality
   ```
3. Commit changes:
   ```bash
   git commit -m "Implement new functionality"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/new-functionality
   ```
5. Submit a Pull Request.

---

## 🙏 Acknowledgments

- Inspired by xAI’s advancements in artificial intelligence.
- Built with Streamlit for an efficient user interface.
- Data sourced from data.gov.

---

## 📬 Contact

For inquiries or feedback:
- **GitHub**: [mansityagi01](https://github.com/mansityagi01)
- **Email**: mansityagi472@gmail.com  *(Update with your actual email)*

---

## ⚙️ System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8 GB RAM (16 GB recommended for large datasets)
- **Storage**: At least 1 GB free space for dataset and dependencies

---
