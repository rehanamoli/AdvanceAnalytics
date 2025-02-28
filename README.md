# **Product Demand Forecasting - Streamlit Application**

## **Overview**
This project implements an advanced demand forecasting system using time series techniques, including **Prophet** and **XGBoost**. The application provides an interactive **Streamlit** interface to visualize and compare forecast results dynamically.

## **Features**
- Load and clean historical product demand data.
- Forecast demand using **Prophet** and **XGBoost** models.
- Display visualizations of actual vs. predicted demand.
- Evaluate model performance using **MAE, RMSE, and MAPE**.
- Interactive filtering options for selecting specific products or categories.

## **Installation**
### **Prerequisites**
Ensure you have **Python 3.8+** installed.

### **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone git@github.com:rehanamoli/AdvanceAnalytics.git
   cd AdvanceAnalytics
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## **Usage**
1. Upload the historical product demand dataset (`Historical Product Demand.csv`).
2. Select forecasting filters (e.g., **Product Code** or **Product Category**).
3. Choose a forecasting model (**Prophet or XGBoost**).
4. Adjust forecast horizon (30-180 days) and optional log transformations.
5. View and analyze the forecasted demand trends and error metrics.

## **File Structure**
```
│── app.py                 # Main Streamlit application
│── requirements.txt       # Required dependencies
│── README.md              # Documentation
│── Historical Product Demand.csv  # Sample dataset
```

## **Technologies Used**
- **Programming Language**: Python
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Forecasting Models**: Prophet, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Evaluation Metrics**: Scikit-learn (MAE, RMSE, MAPE)

## **Future Improvements**
- Incorporate deep learning models (e.g., LSTMs) for enhanced forecasting.
- Improve feature engineering for better accuracy.
- Integrate real-time forecasting with business operations.

## **License**
This project is open-source and available under the **MIT License**.

---

