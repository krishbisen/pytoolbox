import pytest
from pytoolbox.scrapers import SimpleWebScraper

def test_scraper_initialization():
    """Test that scraper initializes with the correct base URL."""
    url = "https://example.com"
    scraper = SimpleWebScraper(url)
    assert scraper.base_url == url
    assert scraper.soup is None