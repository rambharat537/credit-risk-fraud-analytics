#!/usr/bin/env python3
"""
Test Setup Script
Verify that all essential packages are working correctly.
"""

import sys
import os

def test_imports():
    """Test importing essential packages."""
    print("🧪 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ Pandas:", pd.__version__)
    except ImportError as e:
        print("❌ Pandas import failed:", e)
        return False
    
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ NumPy import failed:", e)
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib: OK")
    except ImportError as e:
        print("❌ Matplotlib import failed:", e)
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn: OK")
    except ImportError as e:
        print("❌ Seaborn import failed:", e)
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("✅ Scikit-learn: OK")
    except ImportError as e:
        print("❌ Scikit-learn import failed:", e)
        return False
    
    try:
        import xgboost as xgb
        print("✅ XGBoost:", xgb.__version__)
    except ImportError as e:
        print("❌ XGBoost import failed:", e)
        return False
    
    return True

def create_sample_data():
    """Create a small sample dataset."""
    print("\n📊 Creating sample data...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample credit data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(40, 12, n_samples).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'credit_score': np.random.normal(680, 80, n_samples).astype(int),
            'loan_amount': np.random.lognormal(10, 0.7, n_samples),
            'default': np.random.binomial(1, 0.1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure data directory exists
        os.makedirs('data/sample', exist_ok=True)
        
        # Save sample data
        df.to_csv('data/sample/test_credit_data.csv', index=False)
        
        print(f"✅ Sample data created: {df.shape}")
        print(f"   Default rate: {df['default'].mean():.2%}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return None

def simple_analysis(df):
    """Run a simple analysis."""
    print("\n📈 Running simple analysis...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Basic statistics
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📊 Average age: {df['age'].mean():.1f}")
        print(f"📊 Average income: ${df['income'].mean():,.0f}")
        print(f"📊 Default rate: {df['default'].mean():.2%}")
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(df['age'], bins=20, alpha=0.7, color='skyblue')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(df['credit_score'], bins=20, alpha=0.7, color='lightgreen')
        plt.title('Credit Score Distribution')
        plt.xlabel('Credit Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('dashboards/exports', exist_ok=True)
        plt.savefig('dashboards/exports/test_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Simple analysis completed")
        print("📁 Plot saved to: dashboards/exports/test_analysis.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Credit Risk Analytics - Setup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages failed to import.")
        print("💡 Try running: simple_install.bat")
        return 1
    
    print("\n✅ All essential packages imported successfully!")
    
    # Create sample data
    df = create_sample_data()
    if df is None:
        return 1
    
    # Run simple analysis
    if not simple_analysis(df):
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 Setup test completed successfully!")
    print("✅ Your environment is ready for financial analytics")
    print("\n🚀 Next steps:")
    print("   1. Run full analysis: python run_analysis.py")
    print("   2. Start API server: python src\\api\\fraud_detection_api.py")
    print("   3. Open Jupyter: jupyter notebook")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to continue...")
    sys.exit(exit_code)