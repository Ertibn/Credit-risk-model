"""
Feature Engineering Pipeline for Credit Risk Scoring

This module implements a reproducible feature engineering pipeline
that transforms raw transaction data into model-ready features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMCalculator:
    """Calculate Recency, Frequency, Monetary metrics for customers."""
    
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
    
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with CustomerId, TransactionStartTime, Amount
            
        Returns
        -------
        pd.DataFrame
            RFM metrics per customer
        """
        if self.snapshot_date is None:
            self.snapshot_date = pd.to_datetime(df['TransactionStartTime']).max()
        
        df_copy = df.copy()
        df_copy['TransactionStartTime'] = pd.to_datetime(df_copy['TransactionStartTime'])
        
        rfm = df_copy.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).reset_index()
        
        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        
        logger.info(f'RFM calculated for {len(rfm)} customers')
        return rfm


class FeatureEngineer:
    """Engineer features from transaction data."""
    
    def __init__(self):
        self.snapshot_date = None
        self.rfm_calculator = None
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform raw data into engineered features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data
            
        Returns
        -------
        pd.DataFrame
            Engineered features
        """
        df_features = self._create_time_features(df)
        df_features = self._create_customer_features(df_features)
        df_features = self._encode_categorical(df_features)
        
        logger.info(f'Feature engineering complete: {df_features.shape}')
        return df_features
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        df_copy = df.copy()
        df_copy['TransactionStartTime'] = pd.to_datetime(df_copy['TransactionStartTime'])
        
        df_copy['TransactionHour'] = df_copy['TransactionStartTime'].dt.hour
        df_copy['TransactionDay'] = df_copy['TransactionStartTime'].dt.day
        df_copy['TransactionMonth'] = df_copy['TransactionStartTime'].dt.month
        df_copy['TransactionYear'] = df_copy['TransactionStartTime'].dt.year
        df_copy['TransactionDayOfWeek'] = df_copy['TransactionStartTime'].dt.dayofweek
        
        return df_copy
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction features by customer."""
        df_copy = df.copy()
        
        customer_features = df_copy.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'Value': ['sum', 'mean'],
            'FraudResult': 'sum'
        }).reset_index()
        
        customer_features.columns = [
            'CustomerId',
            'TotalAmount', 'AvgAmount', 'StdAmount', 'MinAmount', 'MaxAmount', 'TxnCount',
            'TotalValue', 'AvgValue',
            'FraudCount'
        ]
        
        customer_features['FraudRate'] = (
            customer_features['FraudCount'] / customer_features['TxnCount']
        )
        customer_features['AvgStdAmount'] = customer_features['StdAmount'].fillna(0)
        
        return customer_features
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df_copy = df.copy()
        
        # One-hot encoding for high-cardinality features would go here
        # For now, keep numerical features
        
        return df_copy


class CreditRiskPipeline:
    """End-to-end feature engineering pipeline."""
    
    def __init__(self):
        self.rfm_calculator = RFMCalculator()
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'CreditRiskPipeline':
        """
        Fit the pipeline on training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data
            
        Returns
        -------
        self
        """
        logger.info('Fitting feature engineering pipeline...')
        
        # Calculate RFM
        self.rfm_df = self.rfm_calculator.calculate_rfm(df)
        
        # Engineer features
        self.feature_df = self.feature_engineer.fit_transform(df)
        
        # Fit scaler on numerical features
        numerical_cols = [
            'TotalAmount', 'AvgAmount', 'TxnCount', 'FraudRate'
        ]
        self.scaler.fit(self.feature_df[numerical_cols])
        
        self.is_fitted = True
        logger.info('Pipeline fitting complete')
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data
            
        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError('Pipeline must be fitted before transform')
        
        # Calculate RFM
        rfm_df = self.rfm_calculator.calculate_rfm(df)
        
        # Engineer features
        feature_df = self.feature_engineer.fit_transform(df)
        
        # Scale numerical features
        numerical_cols = [
            'TotalAmount', 'AvgAmount', 'TxnCount', 'FraudRate'
        ]
        feature_df[numerical_cols] = self.scaler.transform(
            feature_df[numerical_cols]
        )
        
        # Merge with RFM
        result_df = feature_df.merge(rfm_df, on='CustomerId', how='left')
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data
            
        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        self.fit(df)
        
        # Scale numerical features
        numerical_cols = [
            'TotalAmount', 'AvgAmount', 'TxnCount', 'FraudRate'
        ]
        self.feature_df[numerical_cols] = self.scaler.transform(
            self.feature_df[numerical_cols]
        )
        
        # Merge with RFM
        result_df = self.feature_df.merge(self.rfm_df, on='CustomerId', how='left')
        
        return result_df


def create_pipeline() -> CreditRiskPipeline:
    """
    Create and return the credit risk feature engineering pipeline.
    
    Returns
    -------
    CreditRiskPipeline
        Configured pipeline
    """
    return CreditRiskPipeline()
