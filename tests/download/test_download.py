import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import shutil
from data_juicer.download.wikipedia import (
    get_wikipedia_urls, download_wikipedia
)

class TestDownload(unittest.TestCase):
    def setUp(self):
        # Creates a temporary directory that persists until you delete it
        self.temp_dir = tempfile.mkdtemp(prefix='dj_test_')

    def tearDown(self):
        # Clean up the temporary directory after each test
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_wikipedia_urls(self):
        dump_date = "20241101"
        expected_urls = [
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2",
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream2.xml-p41243p151573.bz2",
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream3.xml-p311574p311329.bz2"
        ]
        
        with patch('requests.get') as mock_get:
            # Mock the response from Wikipedia API
            mock_response = MagicMock()
            mock_response.text = "some HTML containing the dump files"
            mock_get.return_value = mock_response
            
            urls = get_wikipedia_urls(dump_date=dump_date)
            
            # Verify the function made the correct API call
            mock_get.assert_called_once_with(
                f"https://dumps.wikimedia.org/enwiki/{dump_date}/")
            
            # Verify returned URLs
            assert len(urls) > 3
            assert urls[0] == expected_urls[0]
            assert urls[1] == expected_urls[1]
            assert urls[2] == expected_urls[2]

    @patch('data_juicer.download.wikipedia.get_wikipedia_urls')
    @patch('data_juicer.download.wikipedia.download_file')
    @patch('data_juicer.download.wikipedia.process_wiki_dump')
    def test_wikipedia_download(self, mock_process, mock_download, mock_get_urls):
        dump_date = "20241101"
        url_limit = 1
        item_limit = 50

        # Mock the URLs returned
        mock_urls = [
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2"
        ]
        mock_get_urls.return_value = mock_urls

        # Mock the download process
        mock_download.return_value = "/tmp/mock_downloaded_file.bz2"

        # Mock the processing result
        mock_df = MagicMock()
        mock_df.take.return_value = [{"text": f"Article {i}"} for i in range(10)]
        mock_process.return_value = mock_df

        # Run the function
        wiki_df = download_wikipedia(
            self.temp_dir, 
            dump_date=dump_date, 
            url_limit=url_limit, 
            item_limit=item_limit
        )

        # Verify the calls
        mock_get_urls.assert_called_once_with(dump_date=dump_date)
        mock_download.assert_called_once_with(
            mock_urls[0], 
            os.path.join(self.temp_dir, os.path.basename(mock_urls[0]))
        )
        mock_process.assert_called_once()

        # Verify the result
        sample = wiki_df.take(10)
        assert len(sample) == 10
        
        # Verify the mocks were used correctly
        mock_df.take.assert_called_once_with(10)


if __name__ == '__main__':
    unittest.main()
