import logging
import sys

import cv2
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from pytoolbox.cli import main
from pytoolbox.cv_helpers import ImageProcessor
from pytoolbox.eda_reporter import EDAReporter
from pytoolbox.logging_config import configure_logging
from pytoolbox.scrapers import SimpleWebScraper


def test_image_processor_pipeline_and_save(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)

    processor = ImageProcessor(source)
    processor.convert_to_grayscale().apply_gaussian_blur().apply_canny_edges()
    processor.save_output(output)

    assert output.exists()


def test_eda_correlation():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "label": ["x", "y", "z"]})
    matrix = EDAReporter(df).calculate_correlation_matrix()
    assert list(matrix.columns) == ["a", "b"]


def test_scraper_fetch_and_heading(monkeypatch):
    class Response:
        text = "<html><body><h2>Hello</h2><h2>World</h2></body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("pytoolbox.scrapers.requests.get", lambda url: Response())
    scraper = SimpleWebScraper("https://example.com")
    scraper.fetch_page()
    assert scraper.extract_heading() == ["Hello", "World"]


def test_logging_configuration_and_cli(caplog, monkeypatch):
    configure_logging(logging.INFO)
    monkeypatch.setattr(sys, "argv", ["pytoolbox", "clamp", "15", "0", "10"])

    with caplog.at_level(logging.INFO, logger="pytoolbox.cli"):
        main()

    assert "result=10" in caplog.text
