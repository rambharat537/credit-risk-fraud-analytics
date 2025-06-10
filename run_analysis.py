#!/usr/bin/env python3
"""
Financial Analytics Pipeline - Main Execution Script
Complete credit risk and fraud detection analysis pipeline.

Author: Ram Bharat Chowdary Moturi
Date: 2025
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Add src directory to path
sys.path.append('src')

# Import custom modules
from utils.data_generator import FinancialDataGenerator
from models.credit_risk_model import CreditRiskModel
from models.fraud_detection_model import FraudDetectionModel

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/financial_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw", "data/processed", "data/sample", "data/external",
        "models", "dashboards/exports", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def generate_sample_data(logger):
    """Generate sample financial datasets."""
    logger.info("🔄 Generating sample financial data...")
    
    try:
        generator = FinancialDataGenerator(random_state=42)
        datasets = generator.save_datasets("data/sample/")
        
        logger.info("✅ Sample data generation completed")
        logger.info(f"📊 Credit data: {datasets['credit_data'].shape}")
        logger.info(f"💳 Transaction data: {datasets['transaction_data'].shape}")
        logger.info(f"👥 Customer data: {datasets['customer_data'].shape}")
        
        return datasets
    
    except Exception as e:
        logger.error(f"❌ Error generating sample data: {e}")
        return None

def train_credit_risk_model(logger):
    """Train credit risk assessment models."""
    logger.info("🏦 Training credit risk models...")
    
    try:
        # Initialize model
        credit_model = CreditRiskModel()
        
        # Load data
        data_path = "data/sample/credit_risk_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"❌ Credit data file not found: {data_path}")
            return None
        
        df = credit_model.load_data(data_path)
        if df is None:
            return None
        
        # Preprocess and train
        X, y = credit_model.preprocess_data(df)
        results = credit_model.train_models(X, y)
        
        # Generate plots
        credit_model.plot_model_comparison()
        
        # Save models
        credit_model.save_models("models/")
        
        logger.info("✅ Credit risk models training completed")
        
        # Display results
        for model_name, result in results.items():
            if 'auc_score' in result:
                logger.info(f"📈 {model_name} AUC Score: {result['auc_score']:.4f}")
        
        return credit_model
    
    except Exception as e:
        logger.error(f"❌ Error training credit risk models: {e}")
        return None

def train_fraud_detection_model(logger):
    """Train fraud detection models."""
    logger.info("🔍 Training fraud detection models...")
    
    try:
        # Initialize model
        fraud_model = FraudDetectionModel()
        
        # Load data
        data_path = "data/sample/transaction_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"❌ Transaction data file not found: {data_path}")
            return None
        
        df = fraud_model.load_data(data_path)
        if df is None:
            return None
        
        # Engineer features and train
        df = fraud_model.engineer_features(df)
        X, y = fraud_model.select_features(df)
        results = fraud_model.train_models(X, y)
        
        # Generate plots
        fraud_model.plot_fraud_analysis()
        
        # Save models
        fraud_model.save_models("models/")
        
        logger.info("✅ Fraud detection models training completed")
        
        # Display results
        for model_name, result in results.items():
            if 'auc_score' in result and result['auc_score'] is not None:
                logger.info(f"📈 {model_name} AUC Score: {result['auc_score']:.4f}")
        
        return fraud_model
    
    except Exception as e:
        logger.error(f"❌ Error training fraud detection models: {e}")
        return None

def run_sample_predictions(credit_model, fraud_model, logger):
    """Run sample predictions to demonstrate model functionality."""
    logger.info("🎯 Running sample predictions...")
    
    try:
        # Sample credit risk prediction
        if credit_model:
            sample_customer = {
                'age': 35,
                'annual_income': 75000,
                'employment_length': 5,
                'credit_score': 720,
                'total_debt': 25000,
                'credit_limit': 15000,
                'num_credit_accounts': 6,
                'loan_amount': 20000,
                'loan_term': 36,
                'debt_to_income_ratio': 0.33,
                'credit_utilization': 0.6,
                'loan_to_income_ratio': 0.27,
                'home_ownership_encoded': 1,
                'loan_purpose_encoded': 0,
                'age_group_encoded': 1,
                'income_group_encoded': 3,
                'credit_score_category_encoded': 3
            }
            
            risk_result = credit_model.predict_risk_score(sample_customer)
            logger.info(f"💰 Credit Risk Assessment:")
            logger.info(f"   Risk Probability: {risk_result['risk_probability']:.4f}")
            logger.info(f"   Risk Category: {risk_result['risk_category']}")
            logger.info(f"   Risk Score: {risk_result['risk_score']}")
        
        # Sample fraud detection
        if fraud_model:
            sample_transaction = {
                'amount': 5000,
                'amount_log': 8.517,
                'hour': 2,
                'day_of_week': 6,
                'is_weekend': 1,
                'is_night': 1,
                'time_diff': 0.5,
                'transactions_last_1h': 3,
                'transactions_last_24h': 15,
                'amount_zscore': 3.2,
                'merchant_risk_score': 0.15,
                'is_international': 1,
                'is_different_state': 0,
                'merchant_category_encoded': 1,
                'payment_method_encoded': 0,
                'merchant_location_encoded': 3
            }
            
            fraud_result = fraud_model.detect_fraud_realtime(sample_transaction)
            logger.info(f"🚨 Fraud Detection Result:")
            logger.info(f"   Is Fraud: {fraud_result['is_fraud']}")
            logger.info(f"   Fraud Probability: {fraud_result['fraud_probability']:.4f}")
            logger.info(f"   Risk Level: {fraud_result['risk_level']}")
            
            if fraud_result['alert_message']:
                logger.info(f"   Alert: {fraud_result['alert_message']}")
        
        logger.info("✅ Sample predictions completed")
        
    except Exception as e:
        logger.error(f"❌ Error running sample predictions: {e}")

def generate_business_report(logger):
    """Generate business summary report."""
    logger.info("📊 Generating business summary report...")
    
    try:
        import pandas as pd
        import json
        
        # Load sample data for reporting
        credit_data = pd.read_csv("data/sample/credit_risk_data.csv")
        transaction_data = pd.read_csv("data/sample/transaction_data.csv")
        customer_data = pd.read_csv("data/sample/customer_data.csv")
        
        # Calculate key metrics
        business_metrics = {
            "report_generated": datetime.now().isoformat(),
            "executive_summary": {
                "total_customers": len(customer_data),
                "total_loans": len(credit_data),
                "total_transactions": len(transaction_data),
                "default_rate": float(credit_data['default'].mean()),
                "fraud_rate": float(transaction_data['is_fraud'].mean()),
                "total_loan_portfolio": float(credit_data['loan_amount'].sum()),
                "total_transaction_volume": float(transaction_data['amount'].sum())
            },
            "risk_metrics": {
                "high_risk_customers": int((credit_data['debt_to_income_ratio'] > 0.4).sum()),
                "high_risk_percentage": float((credit_data['debt_to_income_ratio'] > 0.4).mean() * 100),
                "avg_credit_score": float(credit_data['credit_score'].mean()),
                "avg_debt_to_income": float(credit_data['debt_to_income_ratio'].mean())
            },
            "fraud_insights": {
                "international_transactions": int((transaction_data['merchant_location'] == 'international').sum()),
                "international_fraud_rate": float(transaction_data[transaction_data['merchant_location'] == 'international']['is_fraud'].mean()),
                "night_transactions": int(transaction_data['hour'].isin([0,1,2,3,4,5]).sum()),
                "night_fraud_rate": float(transaction_data[transaction_data['hour'].isin([0,1,2,3,4,5])]['is_fraud'].mean())
            },
            "recommendations": [
                "Implement stricter lending criteria for debt-to-income ratios > 40%",
                "Enhanced monitoring for international transactions",
                "Real-time fraud detection for night-time transactions",
                "Customer segmentation for targeted risk management",
                "Automated model retraining pipeline"
            ]
        }
        
        # Save report
        report_path = "dashboards/exports/business_summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(business_metrics, f, indent=2)
        
        logger.info(f"📋 Business report saved to: {report_path}")
        
        # Print key insights
        logger.info("🎯 Key Business Insights:")
        logger.info(f"   Portfolio Size: ${business_metrics['executive_summary']['total_loan_portfolio']:,.0f}")
        logger.info(f"   Default Rate: {business_metrics['executive_summary']['default_rate']:.2%}")
        logger.info(f"   Fraud Rate: {business_metrics['executive_summary']['fraud_rate']:.4%}")
        logger.info(f"   High-Risk Customers: {business_metrics['risk_metrics']['high_risk_percentage']:.1f}%")
        
        return business_metrics
        
    except Exception as e:
        logger.error(f"❌ Error generating business report: {e}")
        return None

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Financial Analytics Pipeline')
    parser.add_argument('--skip-data-generation', action='store_true', help='Skip data generation step')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--models-only', action='store_true', help='Only train models')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    logger.info("🚀 Starting Financial Analytics Pipeline")
    logger.info("=" * 60)
    
    # Create directories
    create_directories()
    
    # Initialize variables
    credit_model = None
    fraud_model = None
    
    try:
        # Step 1: Generate sample data (unless skipped)
        if not args.skip_data_generation and not args.models_only:
            datasets = generate_sample_data(logger)
            if datasets is None:
                logger.error("❌ Pipeline failed at data generation step")
                return 1
        
        # Step 2: Train models (unless skipped)
        if not args.skip_training:
            credit_model = train_credit_risk_model(logger)
            fraud_model = train_fraud_detection_model(logger)
            
            if credit_model is None and fraud_model is None:
                logger.error("❌ Pipeline failed at model training step")
                return 1
        
        # Step 3: Run sample predictions
        if not args.models_only:
            run_sample_predictions(credit_model, fraud_model, logger)
            
            # Step 4: Generate business report
            generate_business_report(logger)
        
        logger.info("=" * 60)
        logger.info("🎉 Financial Analytics Pipeline completed successfully!")
        logger.info("📁 Check the following directories for outputs:")
        logger.info("   📊 Data: data/sample/")
        logger.info("   🤖 Models: models/")
        logger.info("   📈 Dashboards: dashboards/exports/")
        logger.info("   📝 Logs: logs/")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error in pipeline: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)