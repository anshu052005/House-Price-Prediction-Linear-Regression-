# 🏠 House Price Prediction - Linear Regression Project

## 📊 Project Overview
This project implements a **Linear Regression model** to predict house prices based on key features like square footage, number of bedrooms, and bathrooms. The model achieves **56.43% accuracy (R² score)** and provides a solid foundation for house price prediction.

## 🎯 Key Features

### **Simple Linear Regression Model**
- **Algorithm**: Linear Regression from scikit-learn
- **Features**: Square footage, bedrooms, bathrooms
- **Accuracy**: 56.43% R² score
- **Output**: Predicted house prices in CSV format

### **Data Processing**
- **Data Loading**: Pandas for CSV file handling
- **Feature Engineering**: Basic feature creation and cleaning
- **Outlier Handling**: IQR method for outlier detection
- **Scaling**: StandardScaler for feature normalization
- **Train-Test Split**: 80-20 split for model validation

### **Model Evaluation**
- **R² Score**: 56.43% accuracy
- **Mean Squared Error (MSE)**: Error measurement
- **Root Mean Squared Error (RMSE)**: Standard error metric
- **Mean Absolute Error (MAE)**: Average prediction error

## 📈 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 81.43% | Model accuracy |
| **MSE** | Calculated | Mean squared error |
| **RMSE** | Calculated | Root mean squared error |
| **MAE** | Calculated | Mean absolute error |

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Quick Setup
```bash
# Clone or download the project
# Navigate to project directory
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook house_price_prediction.ipynb
```

## 📖 Usage

### **Running the Model**
1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   - Open `house_price_prediction.ipynb`

3. **Run all cells:**
   - Execute cells sequentially
   - Check the final accuracy results

4. **View results:**
   - Model performance metrics
   - Feature importance
   - Predictions in `house_price_predictions.csv`

### **Expected Output**
```
Linear Regression Model Performance:
Mean Squared Error (MSE): $X,XXX,XXX
Root Mean Squared Error (RMSE): $XX,XXX
Mean Absolute Error (MAE): $XX,XXX
R-squared (R²): 0.5643
R-squared percentage: 56.43%
```

## 🔍 Features Used

### **Primary Features**
- `GrLivArea` (Square footage)
- `BedroomAbvGr` (Number of bedrooms)
- `FullBath` (Number of full bathrooms)
- `HalfBath` (Number of half bathrooms)

### **Engineered Features**
- `TotalBathrooms` (Full + Half bathrooms)
- Scaled features for better model performance

## 📊 Model Details

### **Linear Regression**
- **Algorithm**: `sklearn.linear_model.LinearRegression`
- **Training**: 80% of data
- **Testing**: 20% of data
- **Cross-validation**: Not implemented (basic model)

### **Data Preprocessing**
- **Missing Values**: Handled with mean imputation
- **Outliers**: Removed using IQR method
- **Scaling**: StandardScaler applied to features
- **Feature Selection**: Basic feature engineering

## 📁 Project Structure
```
house-price-prediction/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── house_price_prediction.ipynb        # Main Jupyter notebook
├── house_price_predictions.csv         # Model predictions
├── House train.csv                     # Training dataset
└── House test.csv                      # Test dataset
```

## 📋 Dependencies

### **Core Libraries**
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.0.0` - Machine learning
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization
- `jupyter>=1.0.0` - Notebook environment

## 🎯 Model Performance

### **Current Accuracy: 56.43%**
- **Baseline Model**: Linear Regression
- **Simple Features**: Square footage, bedrooms, bathrooms
- **No Advanced Techniques**: Basic preprocessing only

### **Improvement Opportunities**
- Add more features from the dataset
- Implement feature engineering
- Try ensemble methods
- Use advanced algorithms (Random Forest, XGBoost)

## 📈 Results Interpretation

### **R² Score of 56.43%**
- **Good**: Model explains 56.43% of price variance
- **Room for Improvement**: 43.57% unexplained variance
- **Baseline Performance**: Acceptable for simple linear model

### **Error Metrics**
- **RMSE**: Average prediction error in dollars
- **MAE**: Average absolute error in dollars
- **Lower is Better**: Aim for lower error values

## 🚀 Future Enhancements

### **Immediate Improvements**
- [ ] Add more features (quality, age, location)
- [ ] Implement feature engineering
- [ ] Try different algorithms
- [ ] Add cross-validation

### **Advanced Features**
- [ ] Ensemble methods
- [ ] Hyperparameter tuning
- [ ] Feature selection
- [ ] Model interpretability

## 🛠️ Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Loading Issues**
   - Ensure CSV files are in the same directory
   - Check file names match exactly

3. **Memory Issues**
   - Close other applications
   - Restart Jupyter kernel

### **Performance Tips**
- Use smaller datasets for testing
- Restart kernel if memory issues occur
- Save intermediate results

## 📊 Expected Output Files

### **Generated Files**
- `house_price_predictions.csv` - Model predictions
- Console output with performance metrics
- Visualization plots (if implemented)

### **Sample Output**
```
Id,Predicted_SalePrice
1,185000
2,220000
3,195000
...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the notebook comments
3. Create an issue with detailed description

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Ames Housing Dataset creators
- Scikit-learn development team
- Open source machine learning community

---

**🎯 Current Accuracy: 56.43% R² Score**

This linear regression model provides a solid foundation for house price prediction. While the accuracy is moderate, it demonstrates the basic principles of machine learning and can be enhanced with more advanced techniques. 
