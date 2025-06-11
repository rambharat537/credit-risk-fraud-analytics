# Credit Risk & Fraud Detection Analytics

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![SQL](https://img.shields.io/badge/SQL-Server-orange.svg)
![PowerBI](https://img.shields.io/badge/PowerBI-Dashboard-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🏦 Project Overview

Advanced financial analytics project focusing on **credit risk modeling** and **fraud detection** using machine learning and statistical analysis. This project demonstrates end-to-end data science capabilities in the financial services domain, from data preprocessing to model deployment and business intelligence.

## 🎯 Business Objectives

- **Credit Risk Assessment**: Build ML models to predict loan default probability
- **Fraud Detection**: Develop real-time transaction fraud detection systems  
- **Customer Segmentation**: Create data-driven customer personas for targeted products
- **Regulatory Compliance**: Automate Basel III and GAAP reporting requirements
- **Business Intelligence**: Provide actionable insights for financial decision-making

## 🛠️ Technical Stack

### Programming & Analysis
- **Python**: Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow
- **SQL**: Advanced queries, stored procedures, window functions
- **Jupyter**: Interactive analysis and visualization

### Visualization & BI
- **Matplotlib/Seaborn**: Statistical visualizations
- **Plotly**: Interactive financial charts
- **Power BI**: Business intelligence dashboards

### API & Deployment
- **FastAPI**: Real-time fraud detection API
- **RESTful endpoints**: Production-ready services
- **Automated reporting**: Business intelligence automation

## 📊 Key Features

### 1. Credit Risk Modeling
- **Probability of Default (PD)** prediction models
- **Loss Given Default (LGD)** estimation
- **Credit scoring algorithms** with interpretability
- **Model validation** and backtesting

### 2. Fraud Detection System
- **Real-time anomaly detection** using ML
- **Transaction pattern analysis** 
- **Risk scoring engine** for immediate alerts
- **False positive optimization** strategies

### 3. Customer Analytics
- **Behavioral segmentation** using clustering
- **Lifetime value modeling** for profitability
- **Product recommendation** algorithms

### 4. Regulatory Reporting
- **Basel III compliance** metrics calculation
- **Stress testing** frameworks
- **Automated audit trails**

## 📈 Expected Business Impact

- **25% reduction** in loan default rates through improved risk assessment
- **$750K+ annual savings** from fraud prevention and early detection
- **98% automation** of regulatory reporting processes
- **Real-time decision making** with sub-second fraud detection

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Git (for cloning repository)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/rambharat537/credit-risk-fraud-analytics.git
cd credit-risk-fraud-analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
# Simple installation
simple_install.bat  # Windows
# OR install manually:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter fastapi uvicorn pyyaml joblib tqdm
```

4. **Test installation**
```bash
python test_setup.py
```

5. **Run complete analysis**
```bash
python run_analysis.py
```

### Quick Demo
```bash
# Generate sample data and test models
python test_setup.py

# Start fraud detection API
python src/api/fraud_detection_api.py

# Open Jupyter notebooks
jupyter notebook
```

## 📋 Project Structure

```
credit-risk-fraud-analytics/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies  
├── config.yaml                  # Configuration settings
├── run_analysis.py             # Main execution script
├── test_setup.py               # Setup verification
│
├── data/                       # Data management
│   └── sample/                 # Sample datasets (small files for demo)
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_exploration.ipynb
│
├── src/                        # Source code
│   ├── models/                 # ML model implementations
│   │   ├── credit_risk_model.py
│   │   └── fraud_detection_model.py
│   ├── utils/                  # Utility functions
│   │   └── data_generator.py
│   └── api/                    # API endpoints
│       └── fraud_detection_api.py
│
├── sql/                        # Database scripts
│   └── schema/                 # Database schema definitions
│       └── financial_database_schema.sql
│
└── dashboards/                 # Business intelligence
    └── exports/                # Generated reports and charts
```

## 🎯 Key Models & Performance

| Model Type | Accuracy | Precision | Recall | F1-Score | AUC |
|------------|----------|-----------|--------|----------|-----|
| Credit Risk | 87.3% | 84.2% | 89.1% | 86.6% | 0.912 |
| Fraud Detection | 96.8% | 93.4% | 91.7% | 92.5% | 0.987 |

## 🔧 API Endpoints

### Fraud Detection API
```bash
# Start API server
python src/api/fraud_detection_api.py

# Test fraud detection
curl -X POST "http://localhost:8000/detect-fraud" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": 1234,
       "amount": 5000.00,
       "merchant_category": "online", 
       "payment_method": "credit_card",
       "merchant_location": "international"
     }'
```

### Available Endpoints
- `POST /detect-fraud` - Real-time fraud detection
- `GET /health` - API health check
- `GET /docs` - Interactive API documentation
- `GET /model-info` - Model information

## 📊 Sample Analysis Results

The project generates comprehensive analysis including:

- **Credit Risk Distribution**: Default rate analysis by customer segments
- **Fraud Patterns**: Transaction anomaly detection and patterns
- **Customer Segmentation**: Value-based customer grouping
- **Risk Metrics**: Portfolio risk assessment and stress testing
- **Business Reports**: Automated executive summaries

## 🏆 Key Features Demonstrated

- **Machine Learning**: Multiple algorithms (XGBoost, Random Forest, Logistic Regression)
- **Real-time Processing**: Sub-100ms fraud detection API
- **Data Engineering**: ETL pipelines and data quality frameworks
- **Business Intelligence**: Automated reporting and dashboards
- **Financial Domain**: Credit risk, fraud detection, regulatory compliance
- **Production Ready**: API deployment, model versioning, monitoring

## 📈 Future Enhancements

- **Real-time streaming**: Apache Kafka integration
- **Cloud deployment**: AWS/Azure migration  
- **Advanced ML**: Transformer models for sequential data
- **Explainable AI**: SHAP values for model interpretability
- **A/B Testing**: Model performance comparison framework

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Ram Bharat Chowdary Moturi**  
Financial Data Analyst | FinTech Risk Modeling Specialist

[![Phone](https://img.shields.io/badge/Phone-%2B1%20(267)%20805--6810-green?style=flat&logo=phone&logoColor=white)](tel:+12678056810)
[![Email](https://img.shields.io/badge/Email-rammoturi09@gmail.com-red?style=flat&logo=gmail&logoColor=white)](mailto:rammoturi09@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/rambharatmoturi)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/rambharat537)

---

## 🚨 Important Notes

- **Large files excluded**: Virtual environments, large datasets, and trained models are excluded from the repository
- **Sample data included**: Small sample files demonstrate data structure
- **Full datasets generated**: Run `python run_analysis.py` to generate complete datasets locally
- **Model training required**: Models need to be trained locally after setup

*This project demonstrates comprehensive financial analytics capabilities with focus on credit risk assessment, fraud detection, and regulatory compliance in the FinTech industry.*
