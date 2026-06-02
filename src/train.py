"""
Model Training and Experiment Tracking

Train multiple models and track experiments with MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate models with MLflow tracking."""
    
    def __init__(self, experiment_name: str = "Credit Risk Modeling"):
        """
        Initialize model trainer.
        
        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'is_high_risk',
                     test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Prepare data for model training.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data with features and target
        target_col : str
            Target column name
        test_size : float
            Test set fraction
        random_state : int
            Random state for reproducibility
            
        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        logger.info(f'Preparing data for modeling...')
        
        # Separate features and target
        X = df.drop(columns=[target_col, 'CustomerId'], errors='ignore')
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f'Data split: {len(X_train)} train, {len(X_test)} test')
        logger.info(f'Target distribution - Train: {y_train.value_counts().to_dict()}')
        logger.info(f'Target distribution - Test: {y_test.value_counts().to_dict()}')
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test,
                                 C: float = 1.0) -> dict:
        """Train Logistic Regression model."""
        logger.info('\n=== Training Logistic Regression ===')
        
        with mlflow.start_run(run_name='LogisticRegression'):
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(C=C, random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log parameters
            mlflow.log_params({'C': C, 'model_type': 'LogisticRegression'})
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, 'model')
            
            logger.info(f'Metrics: {metrics}')
            
            return {'model': model, 'metrics': metrics, 'scaler': scaler}
    
    def train_random_forest(self, X_train, X_test, y_train, y_test,
                           n_estimators: int = 100) -> dict:
        """Train Random Forest model."""
        logger.info('\n=== Training Random Forest ===')
        
        with mlflow.start_run(run_name='RandomForest'):
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log parameters
            mlflow.log_params({
                'n_estimators': n_estimators,
                'model_type': 'RandomForest'
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, 'model')
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f'Metrics: {metrics}')
            logger.info(f'Top features: {feature_importance.head().to_dict()}')
            
            return {'model': model, 'metrics': metrics, 'feature_importance': feature_importance}
    
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test,
                               n_estimators: int = 100) -> dict:
        """Train Gradient Boosting model."""
        logger.info('\n=== Training Gradient Boosting ===')
        
        with mlflow.start_run(run_name='GradientBoosting'):
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log parameters
            mlflow.log_params({
                'n_estimators': n_estimators,
                'model_type': 'GradientBoosting'
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, 'model')
            
            logger.info(f'Metrics: {metrics}')
            
            return {'model': model, 'metrics': metrics, 'scaler': scaler}
    
    def compare_models(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        """
        Train and compare multiple models.
        
        Parameters
        ----------
        X_train, X_test, y_train, y_test
            Training and test data
            
        Returns
        -------
        pd.DataFrame
            Model comparison results
        """
        logger.info('=== MODEL COMPARISON ===')
        
        results = []
        
        # Logistic Regression
        lr_result = self.train_logistic_regression(X_train, X_test, y_train, y_test)
        results.append({
            'Model': 'Logistic Regression',
            **lr_result['metrics']
        })
        
        # Random Forest
        rf_result = self.train_random_forest(X_train, X_test, y_train, y_test)
        results.append({
            'Model': 'Random Forest',
            **rf_result['metrics']
        })
        
        # Gradient Boosting
        gb_result = self.train_gradient_boosting(X_train, X_test, y_train, y_test)
        results.append({
            'Model': 'Gradient Boosting',
            **gb_result['metrics']
        })
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        
        logger.info('\n=== MODEL COMPARISON RESULTS ===')
        logger.info(f'\n{comparison_df.to_string(index=False)}')
        
        # Find best model
        best_model_idx = comparison_df['f1'].idxmax()
        self.best_model_name = comparison_df.loc[best_model_idx, 'Model']
        
        logger.info(f'\nBest Model: {self.best_model_name}')
        logger.info(f'Best F1 Score: {comparison_df.loc[best_model_idx, "f1"]:.4f}')
        
        return comparison_df


def main():
    """Main training pipeline."""
    logger.info('Starting model training pipeline...')
    
    # Load data
    df = pd.read_csv('data/processed/training_data.csv')
    logger.info(f'Loaded data: {df.shape}')
    
    # Initialize trainer
    trainer = ModelTrainer(experiment_name='Credit Risk Modeling')
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Compare models
    comparison_df = trainer.compare_models(X_train, X_test, y_train, y_test)
    
    # Save results
    comparison_df.to_csv('data/processed/model_comparison.csv', index=False)
    logger.info('Model comparison saved to data/processed/model_comparison.csv')


if __name__ == '__main__':
    main()
