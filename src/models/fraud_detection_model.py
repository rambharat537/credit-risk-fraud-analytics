"""
Fraud Detection Model
Real-time fraud detection using machine learning and anomaly detection.

Author: Ram Bharat Chowdary Moturi
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """Real-time fraud detection system using multiple ML approaches."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize fraud detection model."""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_column = 'is_fraud'
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.fraud_threshold = self.config['models']['fraud_detection']['threshold']['fraud_score']
        
        print("🔍 Fraud Detection Model initialized")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load transaction data for fraud detection."""
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"✅ Transaction data loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for fraud detection."""
        
        data = df.copy()
        
        print("🔧 Engineering fraud detection features...")
        
        # Time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_night'] = data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # Amount-based features
        data['amount_log'] = np.log1p(data['amount'])
        
        # Customer behavior features (sorted by customer and time)
        data = data.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        
        # Time since last transaction
        data['time_diff'] = data.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 3600
        data['time_diff'].fillna(24, inplace=True)  # First transaction assumption
        
        # Transaction velocity features
        data['transactions_last_1h'] = data.groupby('customer_id').rolling('1H', on='timestamp').size().values - 1
        data['transactions_last_24h'] = data.groupby('customer_id').rolling('24H', on='timestamp').size().values - 1
        
        # Amount statistics per customer
        customer_stats = data.groupby('customer_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
        customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount', 'customer_total_transactions']
        data = data.merge(customer_stats, on='customer_id', how='left')
        
        # Amount deviation from customer's normal behavior
        data['amount_zscore'] = (data['amount'] - data['customer_avg_amount']) / (data['customer_std_amount'] + 0.01)
        data['amount_zscore'].fillna(0, inplace=True)
        
        # Merchant category risk score (based on fraud rates)
        merchant_fraud_rates = data.groupby('merchant_category')[self.target_column].mean().to_dict()
        data['merchant_risk_score'] = data['merchant_category'].map(merchant_fraud_rates)
        
        # Location-based features
        data['is_international'] = (data['merchant_location'] == 'international').astype(int)
        data['is_different_state'] = (data['merchant_location'] == 'different_state').astype(int)
        
        # Encode categorical variables
        categorical_features = ['merchant_category', 'payment_method', 'merchant_location']
        
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                data[f'{feature}_encoded'] = le.fit_transform(data[feature].astype(str))
                self.encoders[feature] = le
        
        return data
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Select features for fraud detection model."""
        
        feature_columns = [
            'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'time_diff', 'transactions_last_1h', 'transactions_last_24h',
            'amount_zscore', 'merchant_risk_score', 'is_international', 'is_different_state',
            'merchant_category_encoded', 'payment_method_encoded', 'merchant_location_encoded'
        ]
        
        # Filter features that exist in the data
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features]
        y = df[self.target_column]
        
        self.feature_names = available_features
        
        print(f"✅ Features selected: {len(available_features)} features")
        print(f"🚨 Fraud rate: {y.mean():.4f}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train fraud detection models."""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        results = {}
        
        print("🚀 Training fraud detection models...")
        
        # 1. Isolation Forest (Unsupervised Anomaly Detection)
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=0.1,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train_scaled)
        self.models['isolation_forest'] = iso_forest
        
        # 2. Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        self.models['logistic_regression'] = lr_model
        
        # 3. XGBoost
        print("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluate models
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name}...")
            
            if name == 'isolation_forest':
                # Isolation Forest returns -1 for outliers, 1 for inliers
                anomaly_scores = model.decision_function(X_test_scaled)
                y_pred = (anomaly_scores < 0).astype(int)  # Outliers as fraud
                y_pred_proba = 1 / (1 + np.exp(anomaly_scores))  # Convert to probabilities
            else:
                if name == 'logistic_regression':
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            if name != 'isolation_forest':  # AUC not meaningful for unsupervised
                auc_score = roc_auc_score(y_test, y_pred_proba)
                print(f"AUC Score: {auc_score:.4f}")
            
            print(f"Classification Report:")
            print(classification_report(y_test, y_pred))
            
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score if name != 'isolation_forest' else None
            }
        
        self.results = results
        return results
    
    def detect_fraud_realtime(self, transaction: Dict, model_name: str = 'xgboost') -> Dict:
        """Real-time fraud detection for a single transaction."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction])
        
        # Note: In production, you'd need to maintain customer history
        # This is a simplified version
        
        # Select features
        X = df[self.feature_names]
        
        # Scale features
        if model_name in ['logistic_regression', 'isolation_forest']:
            X_scaled = self.scalers['standard'].transform(X)
            
            if model_name == 'isolation_forest':
                anomaly_score = model.decision_function(X_scaled)[0]
                fraud_probability = 1 / (1 + np.exp(anomaly_score))
                is_fraud = anomaly_score < 0
            else:
                fraud_probability = model.predict_proba(X_scaled)[0, 1]
                is_fraud = fraud_probability > self.fraud_threshold
        else:
            fraud_probability = model.predict_proba(X)[0, 1]
            is_fraud = fraud_probability > self.fraud_threshold
        
        # Determine risk level
        if fraud_probability < 0.1:
            risk_level = "Low"
        elif fraud_probability < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Generate alert
        alert_message = ""
        if is_fraud:
            alert_message = f"🚨 FRAUD ALERT: High risk transaction detected (Score: {fraud_probability:.3f})"
        elif fraud_probability > 0.3:
            alert_message = f"⚠️ WARNING: Suspicious transaction (Score: {fraud_probability:.3f})"
        
        return {
            'is_fraud': is_fraud,
            'fraud_probability': fraud_probability,
            'risk_level': risk_level,
            'fraud_score': int(fraud_probability * 1000),
            'alert_message': alert_message,
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def plot_fraud_analysis(self):
        """Plot fraud detection analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves (excluding Isolation Forest)
        ax1 = axes[0, 0]
        for name, result in self.results.items():
            if result['auc_score'] is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                auc_score = result['auc_score']
                ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Fraud Detection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confusion Matrix for best model
        ax2 = axes[0, 1]
        best_model = 'xgboost'  # Assume XGBoost is best
        cm = confusion_matrix(self.y_test, self.results[best_model]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'Confusion Matrix - {best_model}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # Feature Importance (XGBoost)
        ax3 = axes[1, 0]
        if 'xgboost' in self.models:
            feature_importance = self.models['xgboost'].feature_importances_
            indices = np.argsort(feature_importance)[-10:]
            
            ax3.barh(range(len(indices)), feature_importance[indices])
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([self.feature_names[i] for i in indices])
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 10 Features - Fraud Detection')
        
        # Fraud Score Distribution
        ax4 = axes[1, 1]
        fraud_scores = self.results['xgboost']['probabilities']
        fraud_labels = self.y_test
        
        ax4.hist(fraud_scores[fraud_labels == 0], bins=50, alpha=0.7, label='Legitimate', density=True)
        ax4.hist(fraud_scores[fraud_labels == 1], bins=50, alpha=0.7, label='Fraud', density=True)
        ax4.axvline(self.fraud_threshold, color='red', linestyle='--', label='Threshold')
        ax4.set_xlabel('Fraud Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Fraud Score Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboards/exports/fraud_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, output_dir: str = "models/"):
        """Save fraud detection models."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/fraud_{name}_model.pkl")
        
        # Save scalers and encoders
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{output_dir}/fraud_{name}_scaler.pkl")
        
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, f"{output_dir}/fraud_{name}_encoder.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, f"{output_dir}/fraud_feature_names.pkl")
        
        print(f"✅ Fraud detection models saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    print("🔍 Fraud Detection Model Training")
    
    # Initialize model
    model = FraudDetectionModel()
    
    # Load data
    data_path = "data/sample/transaction_data.csv"
    
    try:
        # Load and preprocess data
        df = model.load_data(data_path)
        if df is not None:
            # Engineer features
            df = model.engineer_features(df)
            
            # Select features
            X, y = model.select_features(df)
            
            # Train models
            results = model.train_models(X, y)
            
            # Plot analysis
            model.plot_fraud_analysis()
            
            # Save models
            model.save_models()
            
            # Example real-time detection
            sample_transaction = {
                'amount': 5000,
                'amount_log': np.log1p(5000),
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
            
            fraud_result = model.detect_fraud_realtime(sample_transaction)
            print(f"\n🎯 Sample Fraud Detection:")
            print(f"Is Fraud: {fraud_result['is_fraud']}")
            print(f"Fraud Probability: {fraud_result['fraud_probability']:.4f}")
            print(f"Risk Level: {fraud_result['risk_level']}")
            print(f"Alert: {fraud_result['alert_message']}")
            
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_path}")
        print("💡 Please run data_generator.py first to create sample data")