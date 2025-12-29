"""
Feature preprocessing module.

This module provides utilities for scaling
training and testing feature sets.
"""

from sklearn.preprocessing import StandardScaler


def preprocess_features(features_train, features_test):
    """
    Scale training and testing features using StandardScaler.

    Args:
        features_train: Training feature set.
        features_test: Testing feature set.

    Returns:
        tuple: Scaled training features, scaled testing features, fitted scaler.
    """
    scaler = StandardScaler()

    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    return features_train_scaled, features_test_scaled, scaler
