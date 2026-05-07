"""
End-to-end integration test to verify the pipeline runs correctly.
This also ensures we get high code coverage on the CI server.
"""

import pathlib
from unittest.mock import patch

import pytest


def test_full_pipeline_e2e():
    """
    Runs the entire pipeline end-to-end using mocked parameters to speed up execution
    and verify that all modules integrate seamlessly.
    Requires the real dataset at data/raw/telco_churn.csv.
    """
    import yaml

    # Check that the real dataset exists before attempting to run
    with open("configs/params.yaml") as f:
        _params = yaml.safe_load(f)
    raw_path = pathlib.Path(_params["data"]["raw_path"])
    if not raw_path.exists():
        pytest.skip(f"Real dataset not found at {raw_path}. Skipping E2E test.")

    from src.data import prepare, preprocess
    from src.evaluation import evaluate
    from src.features import featurize
    from src.training import train

    # We patch the load_params function in each module to reduce Optuna trials for speed
    def mock_load_params(*args, **kwargs):
        # Read the real params
        with open("configs/params.yaml") as f:
            p = yaml.safe_load(f)

        # Override for speed during tests
        p["training"]["n_optuna_trials"] = 1
        p["training"]["cv_folds"] = 2
        return p

    with patch("src.data.prepare.load_params", side_effect=mock_load_params), \
         patch("src.data.preprocess.load_params", side_effect=mock_load_params), \
         patch("src.features.featurize.load_params", side_effect=mock_load_params), \
         patch("src.training.train.load_params", side_effect=mock_load_params), \
         patch("src.evaluation.evaluate.load_params", side_effect=mock_load_params):

        # Run Data Prepare
        prepare.main()

        # Run Data Preprocess
        preprocess.main()

        # Run Featurize
        featurize.main()

        # Run Training
        train.main()

        # Run Evaluation — call helpers directly to avoid sys.exit() in main()
        params = mock_load_params()
        model = evaluate.load_production_model(params)
        metrics = evaluate.evaluate_model(model, params)
        assert "roc_auc" in metrics
        assert "accuracy" in metrics
