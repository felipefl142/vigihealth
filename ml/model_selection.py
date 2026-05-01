"""Model registry for VigiHealth."""


def get_batch_models(skip_logreg=False, skip_boosting=False, oversampling=False):
    return {
        "LogisticRegression": object(),
        "BalancedRandomForest": object(),
        "LightGBM": object(),
        "XGBoost": object(),
    }
