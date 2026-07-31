import pytest
from pathlib import Path
from pytoolbox.cv_helpers import ImageProcessor

def test_image_processor_file_not_found():
    """Test that initializing with an invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ImageProcessor("non_existent_test_image.jpg")

def test_grayscale_conversion_dimensions():
    """Test grayscale shape transformation if a dummy image exists."""
    sample_path = Path("tests/input_image.jpg")
    if sample_path.exists():
        processor = ImageProcessor(sample_path)
        processor.convert_to_grayscale()
        # Grayscale images should have 2 dimensions (Height, Width)
        assert len(processor.processed_image.shape) == 2