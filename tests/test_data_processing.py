"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import RFMCalculator, FeatureEngineer, CreditRiskPipeline
from src.proxy_target import ProxyTargetCreator


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    
    data = {
        'TransactionId': [f'TXN_{i}' for i in range(30)],
        'CustomerId': ['CUST_001'] * 10 + ['CUST_002'] * 10 + ['CUST_003'] * 10,
        'Amount': np.random.uniform(100, 1000, 30),
        'Value': np.random.uniform(100, 1000, 30),
        'TransactionStartTime': list(dates) * 1 + [dates[0]] * 0,
        'FraudResult': [0] * 28 + [1] * 2
    }
    
    return pd.DataFrame(data)


def test_rfm_calculator(sample_transaction_data):
    """Test RFM calculation."""
    calculator = RFMCalculator()
    rfm = calculator.calculate_rfm(sample_transaction_data)
    
    # Check output shape
    assert rfm.shape[0] == 3, "Should have 3 unique customers"
    assert rfm.shape[1] == 4, "Should have 4 columns (CustomerId, R, F, M)"
    
    # Check column names
    assert 'CustomerId' in rfm.columns
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    
    # Check value ranges
    assert (rfm['Recency'] >= 0).all(), "Recency should be non-negative"
    assert (rfm['Frequency'] > 0).all(), "Frequency should be positive"
    assert (rfm['Monetary'] >= 0).all(), "Monetary should be non-negative"


def test_feature_engineer(sample_transaction_data):
    """Test feature engineering."""
    engineer = FeatureEngineer()
    features = engineer.fit_transform(sample_transaction_data)
    
    # Check output shape
    assert len(features) > 0, "Should generate features"
    assert 'CustomerId' in features.columns, "Should have CustomerId column"
    
    # Check for expected feature columns
    expected_cols = ['TotalAmount', 'AvgAmount', 'TxnCount', 'FraudRate']
    for col in expected_cols:
        assert col in features.columns, f"Should have {col} column"


def test_feature_engineer_columns():
    """Test that feature engineer produces expected columns."""
    # Create test data
    df = pd.DataFrame({
        'TransactionId': ['TXN_1', 'TXN_2', 'TXN_3'],
        'CustomerId': ['CUST_001', 'CUST_001', 'CUST_002'],
        'Amount': [100.0, 200.0, 150.0],
        'Value': [100, 200, 150],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'FraudResult': [0, 0, 1]
    })
    
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df)
    
    # Check that we have customer-level aggregations
    assert len(features) == 2, "Should have 2 customers"
    assert 'TotalAmount' in features.columns
    assert 'FraudRate' in features.columns


def test_credit_risk_pipeline(sample_transaction_data):
    """Test end-to-end credit risk pipeline."""
    pipeline = CreditRiskPipeline()
    features = pipeline.fit_transform(sample_transaction_data)
    
    # Check output
    assert len(features) > 0, "Pipeline should produce output"
    assert 'CustomerId' in features.columns, "Should have CustomerId"
    assert 'Recency' in features.columns, "Should have Recency"
    assert 'Frequency' in features.columns, "Should have Frequency"
    assert 'Monetary' in features.columns, "Should have Monetary"


def test_proxy_target_creator(sample_transaction_data):
    """Test proxy target variable creation."""
    creator = ProxyTargetCreator(n_clusters=3, random_state=42)
    
    target_df = creator.create_target(sample_transaction_data)
    
    # Check output
    assert len(target_df) > 0, "Should create target dataframe"
    assert 'CustomerId' in target_df.columns
    assert 'is_high_risk' in target_df.columns
    
    # Check target values
    assert target_df['is_high_risk'].isin([0, 1]).all(), "Target should be binary"
    assert target_df['is_high_risk'].nunique() <= 2, "Target should have at most 2 classes"


def test_proxy_target_reproducibility():
    """Test that proxy target creation is reproducible."""
    df = pd.DataFrame({
        'TransactionId': ['TXN_1', 'TXN_2', 'TXN_3', 'TXN_4'],
        'CustomerId': ['CUST_001', 'CUST_001', 'CUST_002', 'CUST_002'],
        'Amount': [100.0, 200.0, 50.0, 60.0],
        'Value': [100, 200, 50, 60],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'FraudResult': [0, 0, 0, 0]
    })
    
    # Create target twice
    creator1 = ProxyTargetCreator(random_state=42)
    target1 = creator1.create_target(df)
    
    creator2 = ProxyTargetCreator(random_state=42)
    target2 = creator2.create_target(df)
    
    # Check reproducibility
    assert target1['is_high_risk'].equals(target2['is_high_risk']), \
        "Proxy target should be reproducible with same random_state"


def test_rfm_metrics():
    """Test RFM metric calculations."""
    df = pd.DataFrame({
        'TransactionId': ['TXN_1', 'TXN_2', 'TXN_3'],
        'CustomerId': ['CUST_001', 'CUST_001', 'CUST_001'],
        'Amount': [100.0, 200.0, 150.0],
        'Value': [100, 200, 150],
        'TransactionStartTime': ['2023-01-01', '2023-01-05', '2023-01-10'],
        'FraudResult': [0, 0, 0]
    })
    
    calculator = RFMCalculator()
    rfm = calculator.calculate_rfm(df)
    
    # Check metrics
    assert rfm.loc[0, 'Frequency'] == 3, "Should have 3 transactions"
    assert rfm.loc[0, 'Monetary'] == 450.0, "Should sum to 450"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
