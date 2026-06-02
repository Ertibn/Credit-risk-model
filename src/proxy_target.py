"""
Proxy Target Variable Engineering

Creates a binary high-risk classification using RFM-based customer clustering.
High-risk customers are those with low engagement (low recency, frequency, monetary).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProxyTargetCreator:
    """Create proxy target variable using RFM clustering."""
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize proxy target creator.
        
        Parameters
        ----------
        n_clusters : int, default=3
            Number of clusters for RFM segmentation
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics from transaction data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with CustomerId, TransactionStartTime, Amount
            
        Returns
        -------
        pd.DataFrame
            RFM metrics per customer
        """
        df_copy = df.copy()
        df_copy['TransactionStartTime'] = pd.to_datetime(df_copy['TransactionStartTime'])
        
        snapshot_date = df_copy['TransactionStartTime'].max()
        
        rfm = df_copy.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).reset_index()
        
        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        
        logger.info(f'RFM calculated for {len(rfm)} customers')
        logger.info(f'Recency: mean={rfm["Recency"].mean():.1f}, max={rfm["Recency"].max()}')
        logger.info(f'Frequency: mean={rfm["Frequency"].mean():.1f}, max={rfm["Frequency"].max()}')
        logger.info(f'Monetary: mean={rfm["Monetary"].mean():.2f}, max={rfm["Monetary"].max():.2f}')
        
        return rfm
    
    def fit(self, rfm: pd.DataFrame) -> 'ProxyTargetCreator':
        """
        Fit K-Means clustering on RFM features.
        
        Parameters
        ----------
        rfm : pd.DataFrame
            RFM metrics with columns: CustomerId, Recency, Frequency, Monetary
            
        Returns
        -------
        self
        """
        logger.info('Fitting RFM clustering...')
        
        # Scale RFM features
        rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].values
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        
        # Cluster customers
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        clusters = self.kmeans.fit_predict(rfm_scaled)
        
        # Analyze clusters
        rfm_copy = rfm.copy()
        rfm_copy['Cluster'] = clusters
        
        logger.info('\nCluster Analysis:')
        for cluster_id in range(self.n_clusters):
            cluster_data = rfm_copy[rfm_copy['Cluster'] == cluster_id]
            logger.info(f'\nCluster {cluster_id} (n={len(cluster_data)}):')
            logger.info(f'  Recency: mean={cluster_data["Recency"].mean():.1f}')
            logger.info(f'  Frequency: mean={cluster_data["Frequency"].mean():.1f}')
            logger.info(f'  Monetary: mean={cluster_data["Monetary"].mean():.2f}')
        
        self.is_fitted = True
        return self
    
    def predict_clusters(self, rfm: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster assignment for new RFM data.
        
        Parameters
        ----------
        rfm : pd.DataFrame
            RFM metrics
            
        Returns
        -------
        np.ndarray
            Cluster assignments
        """
        if not self.is_fitted:
            raise ValueError('Fit the model before predicting')
        
        rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].values
        rfm_scaled = self.scaler.transform(rfm_features)
        
        return self.kmeans.predict(rfm_scaled)
    
    def identify_high_risk_cluster(self, rfm: pd.DataFrame) -> int:
        """
        Identify which cluster represents high-risk (low engagement) customers.
        
        High-risk = high recency (inactive), low frequency, low monetary
        
        Parameters
        ----------
        rfm : pd.DataFrame
            RFM metrics
            
        Returns
        -------
        int
            Cluster ID representing high-risk customers
        """
        logger.info('\nIdentifying high-risk cluster...')
        
        clusters = self.predict_clusters(rfm)
        rfm_copy = rfm.copy()
        rfm_copy['Cluster'] = clusters
        
        # Calculate cluster risk scores
        # Higher recency + lower frequency + lower monetary = higher risk
        cluster_scores = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_data = rfm_copy[rfm_copy['Cluster'] == cluster_id]
            
            # Normalize metrics to 0-1 scale for comparison
            recency_score = cluster_data['Recency'].mean() / rfm['Recency'].max()
            frequency_score = 1 - (cluster_data['Frequency'].mean() / rfm['Frequency'].max())
            monetary_score = 1 - (cluster_data['Monetary'].mean() / rfm['Monetary'].max())
            
            # Risk score: high recency + low frequency + low monetary
            risk_score = (recency_score + frequency_score + monetary_score) / 3
            cluster_scores[cluster_id] = risk_score
            
            logger.info(f'Cluster {cluster_id}: risk_score={risk_score:.3f}')
        
        high_risk_cluster = max(cluster_scores, key=cluster_scores.get)
        logger.info(f'\nHigh-risk cluster: {high_risk_cluster}')
        
        return high_risk_cluster
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create is_high_risk binary target variable.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction data
            
        Returns
        -------
        pd.DataFrame
            Original data merged with is_high_risk target
        """
        # Calculate RFM
        rfm = self.create_rfm_features(df)
        
        # Fit clustering
        self.fit(rfm)
        
        # Identify high-risk cluster
        high_risk_cluster = self.identify_high_risk_cluster(rfm)
        
        # Create target variable
        clusters = self.predict_clusters(rfm)
        rfm['is_high_risk'] = (clusters == high_risk_cluster).astype(int)
        
        logger.info(f'\nTarget variable distribution:')
        logger.info(f'  Low-risk (0): {(rfm["is_high_risk"] == 0).sum()} customers ({(rfm["is_high_risk"] == 0).sum() / len(rfm) * 100:.1f}%)')
        logger.info(f'  High-risk (1): {(rfm["is_high_risk"] == 1).sum()} customers ({(rfm["is_high_risk"] == 1).sum() / len(rfm) * 100:.1f}%)')
        
        return rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'is_high_risk']]


def create_proxy_target(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Create proxy target variable from raw transaction data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction data
    n_clusters : int, default=3
        Number of RFM clusters
        
    Returns
    -------
    pd.DataFrame
        Customer-level data with is_high_risk target
    """
    creator = ProxyTargetCreator(n_clusters=n_clusters)
    return creator.create_target(df)
