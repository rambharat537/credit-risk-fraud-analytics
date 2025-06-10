"""
Create small sample data files for GitHub repository.
These show the data structure without being too large.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_small_samples():
    """Create small sample files for repository."""
    
    # Ensure directories exist
    os.makedirs('data/sample', exist_ok=True)
    
    print("📊 Creating small sample files for GitHub...")
    
    # 1. Credit Risk Sample (50 records)
    np.random.seed(42)
    credit_sample = {
        'customer_id': range(1, 51),
        'age': np.random.normal(40, 12, 50).astype(int),
        'annual_income': np.random.lognormal(10.5, 0.5, 50),
        'employment_length': np.random.exponential(5, 50),
        'credit_score': np.random.normal(680, 80, 50).astype(int),
        'total_debt': np.random.lognormal(9, 1, 50),
        'credit_limit': np.random.lognormal(8.5, 0.8, 50),
        'num_credit_accounts': np.random.poisson(8, 50),
        'loan_amount': np.random.lognormal(10, 0.7, 50),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], 50),
        'home_ownership': np.random.choice(['OWN', 'RENT', 'MORTGAGE'], 50),
        'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase'], 50),
        'debt_to_income_ratio': None,
        'credit_utilization': None,
        'loan_to_income_ratio': None,
        'default': np.random.binomial(1, 0.1, 50)
    }
    
    credit_df = pd.DataFrame(credit_sample)
    credit_df['debt_to_income_ratio'] = credit_df['total_debt'] / credit_df['annual_income']
    credit_df['credit_utilization'] = np.minimum(credit_df['total_debt'] / credit_df['credit_limit'], 1.0)
    credit_df['loan_to_income_ratio'] = credit_df['loan_amount'] / credit_df['annual_income']
    
    credit_df.to_csv('data/sample/credit_risk_sample.csv', index=False)
    
    # 2. Transaction Sample (100 records)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(hours=i*6) for i in range(100)]
    
    transaction_sample = {
        'transaction_id': range(1, 101),
        'customer_id': np.random.randint(1, 20, 100),
        'timestamp': timestamps,
        'amount': np.random.lognormal(3, 1.5, 100),
        'merchant_category': np.random.choice(['grocery', 'gas_station', 'restaurant', 'retail', 'online'], 100),
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'online'], 100),
        'merchant_location': np.random.choice(['same_city', 'same_state', 'different_state', 'international'], 100),
        'is_fraud': np.random.binomial(1, 0.02, 100)
    }
    
    transaction_df = pd.DataFrame(transaction_sample)
    transaction_df.to_csv('data/sample/transaction_sample.csv', index=False)
    
    # 3. Customer Sample (25 records)
    customer_sample = {
        'customer_id': range(1, 26),
        'age': np.random.normal(45, 15, 25).astype(int),
        'gender': np.random.choice(['M', 'F'], 25),
        'income': np.random.lognormal(10.8, 0.6, 25),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 25),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 25),
        'city_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], 25),
        'account_tenure_months': np.random.exponential(36, 25).astype(int),
        'num_products': np.random.poisson(2.5, 25),
        'total_relationship_balance': np.random.lognormal(9.5, 1.2, 25),
        'customer_value_score': np.random.uniform(0, 1, 25)
    }
    
    customer_df = pd.DataFrame(customer_sample)
    customer_df.to_csv('data/sample/customer_sample.csv', index=False)
    
    print("✅ Sample files created:")
    print(f"   📋 Credit Risk Sample: {len(credit_df)} records")
    print(f"   💳 Transaction Sample: {len(transaction_df)} records")
    print(f"   👥 Customer Sample: {len(customer_df)} records")
    print(f"   📁 Saved to: data/sample/")

if __name__ == "__main__":
    create_small_samples()