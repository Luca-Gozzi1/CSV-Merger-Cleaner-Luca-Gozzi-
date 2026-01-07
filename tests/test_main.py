"""
Unit tests for the main orchestrator module.

These tests verify that the SupplyChainPipeline class works correctly
for all pipeline modes.

Run with: pytest tests/test_main.py -v

Author: Luca Gozzi
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.main import SupplyChainPipeline


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory structure."""
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    (Path(temp_dir) / "data" / "raw").mkdir(parents=True)
    (Path(temp_dir) / "data" / "processed").mkdir(parents=True)
    (Path(temp_dir) / "models").mkdir(parents=True)
    (Path(temp_dir) / "results" / "figures").mkdir(parents=True)
    
    yield Path(temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_project_dir_with_csv():
    """Create a temporary project directory with a dummy CSV file."""
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    raw_dir = Path(temp_dir) / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (Path(temp_dir) / "data" / "processed").mkdir(parents=True)
    (Path(temp_dir) / "models").mkdir(parents=True)
    (Path(temp_dir) / "results" / "figures").mkdir(parents=True)
    
    # Create a dummy CSV file
    dummy_csv = raw_dir / "dummy_data.csv"
    dummy_csv.write_text("col1,col2\n1,2\n3,4\n")
    
    yield Path(temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        "order date (DateOrders)": pd.date_range("2020-01-01", periods=n),
        "shipping date (DateOrders)": pd.date_range("2020-01-05", periods=n),
        "Days for shipping (real)": np.random.randint(1, 10, n),
        "Days for shipment (scheduled)": np.random.randint(1, 10, n),
        "Shipping Mode": np.random.choice(["Standard Class", "First Class"], n),
        "Market": np.random.choice(["LATAM", "Europe", "Pacific Asia"], n),
        "Order Item Quantity": np.random.randint(1, 10, n),
        "Sales per customer": np.random.uniform(100, 1000, n),
        "Order Item Discount Rate": np.random.uniform(0, 0.3, n),
        "Late_delivery_risk": np.random.randint(0, 2, n),
    })


@pytest.fixture
def pipeline(temp_project_dir):
    """Create a pipeline instance with temp directory."""
    return SupplyChainPipeline(project_dir=temp_project_dir)


@pytest.fixture
def pipeline_with_csv(temp_project_dir_with_csv):
    """Create a pipeline instance with temp directory containing a CSV."""
    return SupplyChainPipeline(project_dir=temp_project_dir_with_csv)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestSupplyChainPipelineInit:
    """Tests for SupplyChainPipeline initialization."""
    
    def test_init_creates_directories(self, temp_project_dir):
        """Should create necessary directories."""
        pipeline = SupplyChainPipeline(project_dir=temp_project_dir)
        
        assert pipeline.results_dir.exists()
        assert pipeline.figures_dir.exists()
        assert pipeline.models_dir.exists()
    
    def test_init_stores_project_dir(self, temp_project_dir):
        """Should store project directory."""
        pipeline = SupplyChainPipeline(project_dir=temp_project_dir)
        
        assert pipeline.project_dir == temp_project_dir
    
    def test_init_with_default_dir(self):
        """Should work with default directory."""
        pipeline = SupplyChainPipeline()
        
        assert pipeline.project_dir is not None
    
    def test_init_sets_data_dir(self, temp_project_dir):
        """Should set data directory correctly."""
        pipeline = SupplyChainPipeline(project_dir=temp_project_dir)
        
        assert pipeline.data_dir == temp_project_dir / "data"


# =============================================================================
# HELPER METHOD TESTS
# =============================================================================

class TestHelperMethods:
    """Tests for pipeline helper methods."""
    
    def test_check_models_exist_false(self, pipeline):
        """Should return False when no models exist."""
        result = pipeline._check_models_exist()
        
        assert result == False
    
    def test_check_models_exist_true(self, pipeline):
        """Should return True when all models exist."""
        # Create dummy model files
        (pipeline.models_dir / "logistic_regression.pkl").touch()
        (pipeline.models_dir / "random_forest.pkl").touch()
        (pipeline.models_dir / "xgboost.pkl").touch()
        
        result = pipeline._check_models_exist()
        
        assert result == True
    
    def test_check_models_exist_partial(self, pipeline):
        """Should return False when only some models exist."""
        # Create only one model file
        (pipeline.models_dir / "logistic_regression.pkl").touch()
        
        result = pipeline._check_models_exist()
        
        assert result == False


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""
    
    def test_generate_report_creates_file(self, pipeline):
        """Should create report file."""
        results = {
            "status": "success",
            "stages": {
                "load": {"status": "success", "rows": 100},
                "validate": {"status": "success"},
            }
        }
        
        pipeline._generate_report(results)
        
        report_path = pipeline.results_dir / "pipeline_report.txt"
        assert report_path.exists()
    
    def test_generate_report_content(self, pipeline):
        """Report should contain key information."""
        results = {
            "status": "success",
            "stages": {
                "load": {"status": "success", "rows": 100},
            }
        }
        
        pipeline._generate_report(results)
        
        report_path = pipeline.results_dir / "pipeline_report.txt"
        content = report_path.read_text()
        
        assert "SUPPLY CHAIN EXPLORER" in content
        assert "PIPELINE REPORT" in content
        assert "success" in content
    
    def test_generate_report_empty_stages(self, pipeline):
        """Should handle empty stages."""
        results = {
            "status": "success",
            "stages": {}
        }
        
        pipeline._generate_report(results)
        
        report_path = pipeline.results_dir / "pipeline_report.txt"
        assert report_path.exists()


# =============================================================================
# DATA LOADING TESTS (with mocking)
# =============================================================================

class TestDataLoading:
    """Tests for data loading with mocking."""
    
    @patch('src.main.DataLoader')
    def test_load_data_calls_loader(self, mock_loader_class, pipeline_with_csv, sample_dataframe):
        """Should call DataLoader correctly."""
        # Setup mock - use .load() not .load_supply_chain_data()
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = sample_dataframe
        mock_loader_class.return_value = mock_loader_instance
        
        result = pipeline_with_csv._load_data()
        
        # Verify DataLoader was instantiated and load() was called
        mock_loader_class.assert_called_once()
        mock_loader_instance.load.assert_called_once()
        assert len(result) == len(sample_dataframe)


# =============================================================================
# VALIDATION TESTS (with mocking)
# =============================================================================

class TestDataValidation:
    """Tests for data validation with mocking."""
    
    @patch('src.main.DataValidator')
    def test_validate_data_returns_dict(self, mock_validator_class, pipeline, sample_dataframe):
        """Should return validation dictionary."""
        mock_validator = Mock()
        mock_report = Mock()
        mock_report.is_valid = True
        mock_validator.validate_all.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        result = pipeline._validate_data(sample_dataframe)
        
        assert isinstance(result, dict)
        assert "is_valid" in result


# =============================================================================
# PREPROCESSING TESTS (with mocking)
# =============================================================================

class TestPreprocessing:
    """Tests for preprocessing with mocking."""
    
    @patch('src.main.DataPreprocessor')
    def test_preprocess_data_returns_dataframe(self, mock_preprocessor_class, pipeline, sample_dataframe):
        """Should return preprocessed DataFrame."""
        # Setup mock - use .preprocess() and return actual DataFrame
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess.return_value = sample_dataframe
        mock_preprocessor_class.return_value = mock_preprocessor
        
        result = pipeline._preprocess_data(sample_dataframe)
        
        assert isinstance(result, pd.DataFrame)
        mock_preprocessor.preprocess.assert_called_once()


# =============================================================================
# FEATURE ENGINEERING TESTS (with mocking)
# =============================================================================

class TestFeatureEngineering:
    """Tests for feature engineering with mocking."""
    
    @patch('src.main.FeatureEngineer')
    def test_engineer_features_returns_dataframe(self, mock_engineer_class, pipeline, sample_dataframe):
        """Should return engineered DataFrame."""
        mock_engineer = Mock()
        mock_engineer.engineer_all.return_value = sample_dataframe
        mock_engineer_class.return_value = mock_engineer
        
        result = pipeline._engineer_features(sample_dataframe)
        
        assert isinstance(result, pd.DataFrame)
        mock_engineer.engineer_all.assert_called_once()


# =============================================================================
# EVALUATION TESTS
# =============================================================================

class TestEvaluation:
    """Tests for model evaluation."""
    
    def test_evaluate_models_no_data(self, pipeline):
        """Should handle missing data gracefully."""
        # No split data exists
        result = pipeline._evaluate_models()
        
        assert result == {}


# =============================================================================
# FULL PIPELINE TESTS (with extensive mocking)
# =============================================================================

class TestFullPipeline:
    """Tests for full pipeline execution."""
    
    @patch('src.main.DataLoader')
    @patch('src.main.DataValidator')
    @patch('src.main.DataPreprocessor')
    @patch('src.main.FeatureEngineer')
    def test_run_full_pipeline_structure(
        self,
        mock_engineer_class,
        mock_preprocessor_class,
        mock_validator_class,
        mock_loader_class,
        pipeline_with_csv,
        sample_dataframe
    ):
        """Should return results dictionary with correct structure."""
        # Setup mocks with CORRECT method names and return actual DataFrame
        mock_loader = Mock()
        mock_loader.load.return_value = sample_dataframe
        mock_loader_class.return_value = mock_loader
        
        mock_validator = Mock()
        mock_report = Mock()
        mock_report.is_valid = True
        mock_validator.validate_all.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess.return_value = sample_dataframe
        mock_preprocessor_class.return_value = mock_preprocessor
        
        mock_engineer = Mock()
        mock_engineer.engineer_all.return_value = sample_dataframe
        mock_engineer_class.return_value = mock_engineer
        
        # Mock model checking to skip training
        with patch.object(pipeline_with_csv, '_check_models_exist', return_value=True):
            with patch.object(pipeline_with_csv, '_evaluate_models', return_value={}):
                result = pipeline_with_csv.run_full_pipeline(retrain=False)
        
        assert "status" in result
        assert "stages" in result
        assert result["status"] == "success"
    
    def test_run_full_pipeline_handles_errors(self, pipeline):
        """Should handle errors gracefully."""
        with patch.object(pipeline, '_load_data', side_effect=FileNotFoundError("Data not found")):
            with pytest.raises(FileNotFoundError):
                pipeline.run_full_pipeline()


# =============================================================================
# COMMAND LINE ARGUMENT TESTS
# =============================================================================

class TestCommandLineInterface:
    """Tests for command line interface."""
    
    def test_main_function_exists(self):
        """Should have callable main function."""
        from src.main import main
        
        assert callable(main)
    
    def test_supply_chain_pipeline_class_exists(self):
        """Should have SupplyChainPipeline class."""
        from src.main import SupplyChainPipeline
        
        assert SupplyChainPipeline is not None