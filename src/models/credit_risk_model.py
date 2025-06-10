"""
Credit Risk Assessment Model
Machine learning model for predicting loan default probability.

Author: Ram Bharat Chowdary Moturi
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModel:
    """Credit risk assessment model using multiple ML algorithms."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the credit risk model."""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_column = 'default'
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        print("🏦 Credit Risk Model initialized")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load credit risk dataset."""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the credit risk data."""
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in numeric_columns:
            if col != self.target_column:
                data[col].fillna(data[col].median(), inplace=True)
        
        for col in categorical_columns:
            data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Feature Engineering
        print("🔧 Performing feature engineering...")
        
        # Debt-to-income ratio
        if 'debt_to_income_ratio' not in data.columns:
            data['debt_to_income_ratio'] = data['total_debt'] / data['annual_income']
        
        # Credit utilization
        if 'credit_utilization' not in data.columns:
            data['credit_utilization'] = np.minimum(data['total_debt'] / data['credit_limit'], 1.0)
        
        # Loan-to-income ratio
        if 'loan_to_income_ratio' not in data.columns:
            data['loan_to_income_ratio'] = data['loan_amount'] / data['annual_income']
        
        # Age groups
        data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 50, 65, 100], 
                                  labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
        
        # Income groups
        data['income_group'] = pd.qcut(data['annual_income'], q=5, 
                                      labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High'])
        
        # Credit score categories
        data['credit_score_category'] = pd.cut(data['credit_score'], 
                                             bins=[0, 580, 670, 740, 800, 850],
                                             labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        
        # Encode categorical variables
        categorical_features = ['home_ownership', 'loan_purpose', 'age_group', 
                              'income_group', 'credit_score_category']
        
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                data[f'{feature}_encoded'] = le.fit_transform(data[feature].astype(str))
                self.encoders[feature] = le
        
        # Select features for modeling
        feature_columns = [
            'age', 'annual_income', 'employment_length', 'credit_score',
            'total_debt', 'credit_limit', 'num_credit_accounts', 'loan_amount',
            'loan_term', 'debt_to_income_ratio', 'credit_utilization',
            'loan_to_income_ratio', 'home_ownership_encoded', 'loan_purpose_encoded',
            'age_group_encoded', 'income_group_encoded', 'credit_score_category_encoded'
        ]
        
        # Filter features that exist in the data
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features]
        y = data[self.target_column]
        
        self.feature_names = available_features
        
        print(f"✅ Features prepared: {len(available_features)} features")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple credit risk models."""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        results = {}
        
        print("🚀 Training credit risk models...")
        
        # 1. Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        self.models['logistic_regression'] = lr_model
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # 3. XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluate all models
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name}...")
            
            if name == 'logistic_regression':
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            results[name] = {
                'auc_score': auc_score,
                'avg_precision': avg_precision,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"AUC Score: {auc_score:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
        
        self.results = results
        return results
    
    def plot_model_comparison(self):
        """Plot model performance comparison."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves
        ax1 = axes[0, 0]
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            auc_score = result['auc_score']
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax2 = axes[0, 1]
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            avg_precision = result['avg_precision']
            ax2.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Feature Importance (Random Forest)
        ax3 = axes[1, 0]
        if 'random_forest' in self.models:
            feature_importance = self.models['random_forest'].feature_importances_
            indices = np.argsort(feature_importance)[-10:]  # Top 10 features
            
            ax3.barh(range(len(indices)), feature_importance[indices])
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([self.feature_names[i] for i in indices])
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 10 Feature Importances (Random Forest)')
        
        # Model Performance Comparison
        ax4 = axes[1, 1]
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        avg_precisions = [self.results[name]['avg_precision'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax4.bar(x - width/2, auc_scores, width, label='AUC Score', alpha=0.8)
        ax4.bar(x + width/2, avg_precisions, width, label='Average Precision', alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Score')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboards/exports/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_risk_score(self, customer_data: Dict, model_name: str = 'xgboost') -> Dict:
        """Predict credit risk score for a single customer."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Convert customer data to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Apply same preprocessing
        # This is a simplified version - in production, you'd want to reuse the exact preprocessing pipeline
        
        # Select features (ensure same order as training)
        X = df[self.feature_names]
        
        # Scale if needed
        if model_name == 'logistic_regression':
            X_scaled = self.scalers['standard'].transform(X)
            risk_probability = model.predict_proba(X_scaled)[0, 1]
        else:
            risk_probability = model.predict_proba(X)[0, 1]
        
        # Determine risk category
        if risk_probability < 0.05:
            risk_category = "Low Risk"
        elif risk_probability < 0.15:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        return {
            'risk_probability': risk_probability,
            'risk_category': risk_category,
            'risk_score': int(risk_probability * 1000),  # Scale to 0-1000
            'model_used': model_name
        }
    
    def save_models(self, output_dir: str = "models/"):
        """Save trained models and preprocessing objects."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/{name}_model.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{output_dir}/{name}_scaler.pkl")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, f"{output_dir}/{name}_encoder.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, f"{output_dir}/feature_names.pkl")
        
        print(f"✅ Models saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    print("🏦 Credit Risk Model Training")
    
    # Initialize model
    model = CreditRiskModel()
    
    # Load data (you'll need to generate it first using data_generator.py)
    data_path = "data/sample/credit_risk_data.csv"
    
    try:
        # Load and preprocess data
        df = model.load_data(data_path)
        if df is not None:
            X, y = model.preprocess_data(df)
            
            # Train models
            results = model.train_models(X, y)
            
            # Plot results
            model.plot_model_comparison()
            
            # Save models
            model.save_models()
            
            # Example prediction
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
            
            risk_assessment = model.predict_risk_score(sample_customer)
            print(f"\n🎯 Sample Risk Assessment:")
            print(f"Risk Probability: {risk_assessment['risk_probability']:.4f}")
            print(f"Risk Category: {risk_assessment['risk_category']}")
            print(f"Risk Score: {risk_assessment['risk_score']}")
            
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_path}")
        print("💡 Please run data_generator.py first to create sample data")
