"""
Financial Data Generator
Generates synthetic financial datasets for credit risk and fraud detection analysis.

Author: Ram Bharat Chowdary Moturi
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict
import os

class FinancialDataGenerator:
    """Generate synthetic financial datasets for analysis."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the data generator with random state for reproducibility."""
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def generate_credit_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic credit risk dataset."""
        
        # Customer demographics
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(40, 12, n_samples).astype(int),
            'annual_income': np.random.lognormal(10.5, 0.5, n_samples),
            'employment_length': np.random.exponential(5, n_samples),
            'credit_score': np.random.normal(680, 80, n_samples).astype(int),
            'total_debt': np.random.lognormal(9, 1, n_samples),
            'credit_limit': np.random.lognormal(8.5, 0.8, n_samples),
            'num_credit_accounts': np.random.poisson(8, n_samples),
            'loan_amount': np.random.lognormal(10, 0.7, n_samples),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'home_ownership': np.random.choice(['OWN', 'RENT', 'MORTGAGE'], n_samples, p=[0.3, 0.4, 0.3]),
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'other'], 
                                           n_samples, p=[0.4, 0.2, 0.2, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic constraints
        df['age'] = np.clip(df['age'], 18, 80)
        df['credit_score'] = np.clip(df['credit_score'], 300, 850)
        df['employment_length'] = np.clip(df['employment_length'], 0, 40)
        
        # Calculate derived features
        df['debt_to_income_ratio'] = df['total_debt'] / df['annual_income']
        df['credit_utilization'] = np.minimum(df['total_debt'] / df['credit_limit'], 1.0)
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
        
        # Generate target variable (default) based on realistic factors
        default_probability = (
            0.02 +  # Base rate
            0.1 * (df['credit_score'] < 600).astype(int) +
            0.05 * (df['debt_to_income_ratio'] > 0.4).astype(int) +
            0.03 * (df['credit_utilization'] > 0.8).astype(int) +
            0.02 * (df['employment_length'] < 1).astype(int)
        )
        
        df['default'] = np.random.binomial(1, np.clip(default_probability, 0, 1), n_samples)
        
        return df
    
    def generate_transaction_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """Generate synthetic transaction data for fraud detection."""
        
        # Generate transaction timestamps
        start_date = datetime.now() - timedelta(days=365)
        timestamps = [start_date + timedelta(
            seconds=random.randint(0, 365*24*3600)
        ) for _ in range(n_samples)]
        
        data = {
            'transaction_id': range(1, n_samples + 1),
            'customer_id': np.random.randint(1, 5000, n_samples),
            'timestamp': timestamps,
            'amount': np.random.lognormal(3, 1.5, n_samples),
            'merchant_category': np.random.choice([
                'grocery', 'gas_station', 'restaurant', 'retail', 'online', 
                'atm', 'pharmacy', 'entertainment', 'travel', 'other'
            ], n_samples),
            'payment_method': np.random.choice(['credit_card', 'debit_card', 'online'], n_samples, p=[0.6, 0.3, 0.1]),
            'merchant_location': np.random.choice(['same_city', 'same_state', 'different_state', 'international'], 
                                                n_samples, p=[0.7, 0.15, 0.1, 0.05])
        }
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate transaction velocity features
        df = df.sort_values(['customer_id', 'timestamp'])
        df['time_since_last_transaction'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['transactions_last_hour'] = df.groupby('customer_id').rolling('1H', on='timestamp').size().values - 1
        df['amount_zscore'] = df.groupby('customer_id')['amount'].transform(lambda x: (x - x.mean()) / x.std())
        
        # Generate fraud labels based on suspicious patterns
        fraud_probability = (
            0.001 +  # Base fraud rate
            0.1 * (df['amount'] > df['amount'].quantile(0.99)).astype(int) +
            0.05 * (df['merchant_location'] == 'international').astype(int) +
            0.03 * (df['hour'].isin([0, 1, 2, 3, 4, 5])).astype(int) +
            0.02 * (df['transactions_last_hour'] > 5).astype(int) +
            0.05 * (abs(df['amount_zscore']) > 3).astype(int)
        )
        
        df['is_fraud'] = np.random.binomial(1, np.clip(fraud_probability, 0, 1), n_samples)
        
        return df
    
    def generate_customer_data(self, n_customers: int = 5000) -> pd.DataFrame:
        """Generate customer profile data for segmentation."""
        
        data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(45, 15, n_customers).astype(int),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'income': np.random.lognormal(10.8, 0.6, n_customers),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                        n_customers, p=[0.3, 0.4, 0.25, 0.05]),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 
                                             n_customers, p=[0.35, 0.5, 0.15]),
            'num_dependents': np.random.poisson(1.5, n_customers),
            'city_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], n_customers, p=[0.4, 0.35, 0.25]),
            'account_tenure_months': np.random.exponential(36, n_customers).astype(int),
            'num_products': np.random.poisson(2.5, n_customers),
            'total_relationship_balance': np.random.lognormal(9.5, 1.2, n_customers),
            'credit_card_limit': np.random.lognormal(8.8, 0.9, n_customers),
            'mortgage_balance': np.random.lognormal(11.5, 0.8, n_customers) * np.random.binomial(1, 0.3, n_customers),
            'investment_balance': np.random.lognormal(9, 1.5, n_customers) * np.random.binomial(1, 0.6, n_customers)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic constraints
        df['age'] = np.clip(df['age'], 18, 80)
        df['num_dependents'] = np.clip(df['num_dependents'], 0, 8)
        df['account_tenure_months'] = np.clip(df['account_tenure_months'], 1, 300)
        df['num_products'] = np.clip(df['num_products'], 1, 10)
        
        # Calculate customer value score
        df['customer_value_score'] = (
            0.3 * (df['total_relationship_balance'] / df['total_relationship_balance'].max()) +
            0.2 * (df['income'] / df['income'].max()) +
            0.2 * (df['account_tenure_months'] / df['account_tenure_months'].max()) +
            0.15 * (df['num_products'] / df['num_products'].max()) +
            0.15 * ((df['investment_balance'] > 0).astype(int))
        )
        
        return df
    
    def save_datasets(self, output_dir: str = "data/sample/"):
        """Generate and save all datasets."""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating synthetic financial datasets...")
        
        # Generate datasets
        credit_data = self.generate_credit_data(10000)
        transaction_data = self.generate_transaction_data(50000)
        customer_data = self.generate_customer_data(5000)
        
        # Save to CSV files
        credit_data.to_csv(f"{output_dir}/credit_risk_data.csv", index=False)
        transaction_data.to_csv(f"{output_dir}/transaction_data.csv", index=False)
        customer_data.to_csv(f"{output_dir}/customer_data.csv", index=False)
        
        print(f"✅ Credit risk data saved: {len(credit_data)} records")
        print(f"✅ Transaction data saved: {len(transaction_data)} records")
        print(f"✅ Customer data saved: {len(customer_data)} records")
        print(f"📁 Files saved to: {output_dir}")
        
        return {
            'credit_data': credit_data,
            'transaction_data': transaction_data,
            'customer_data': customer_data
        }

if __name__ == "__main__":
    # Generate and save datasets
    generator = FinancialDataGenerator()
    datasets = generator.save_datasets()
    
    print("\n📊 Dataset Summary:")
    print(f"Credit Risk Dataset: {datasets['credit_data'].shape}")
    print(f"Transaction Dataset: {datasets['transaction_data'].shape}")
    print(f"Customer Dataset: {datasets['customer_data'].shape}")
    
    print("\n🎯 Default Rate:", datasets['credit_data']['default'].mean())
    print("🚨 Fraud Rate:", datasets['transaction_data']['is_fraud'].mean())
