"""Pytest configuration for image classification tests."""

import sys
from pathlib import Path

import pytest

# Add project to path for imports
project_path = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_path))


def _check_numpy_torch_compatibility():
    """Check if numpy and torchvision are compatible."""
    try:
        from PIL import Image
        from torchvision import transforms

        # Try a simple transform that uses numpy
        img = Image.new("RGB", (10, 10), color="red")
        transform = transforms.ToTensor()
        _ = transform(img)
        return True
    except RuntimeError as e:
        if "Numpy is not available" in str(e) or "_ARRAY_API not found" in str(e):
            return False
        raise


# Check compatibility once at import time
NUMPY_TORCH_COMPATIBLE = _check_numpy_torch_compatibility()


@pytest.fixture(scope="session")
def project_dir():
    """Return the project directory path."""
    return project_path


@pytest.fixture(scope="session")
def numpy_torch_compatible():
    """Return whether numpy/torch are compatible."""
    return NUMPY_TORCH_COMPATIBLE


def pytest_collection_modifyitems(config, items):
    """Skip tests that require numpy/torch compatibility when not available."""
    if NUMPY_TORCH_COMPATIBLE:
        return

    skip_marker = pytest.mark.skip(
        reason="NumPy and PyTorch/TorchVision have compatibility issues in this environment"
    )

    for item in items:
        # Skip tests in classes that need transforms
        if any(
            marker in str(item.nodeid)
            for marker in [
                "TestTransforms",
                "TestImageClassificationDataset",
                "TestDataLoaders",
                "TestCIFAR10Transforms",
                "TestCIFAR10DataLoaders",
                "TestImageClassifier",
                "TestTrainingFunctions",
                "TestFullTraining",
                "TestPrepareData",
                "TestEvaluate",
            ]
        ):
            item.add_marker(skip_marker)
