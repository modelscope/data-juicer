import unittest
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
        urls = get_wikipedia_urls(dump_date=dump_date)
        assert len(urls) > 3
        assert urls[0] == "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2"
        assert urls[1] == "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream2.xml-p41243p151573.bz2"
        assert urls[2] == "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream3.xml-p151574p311329.bz2"

    def test_wikipedia_download(self):
        dump_date = "20241101"
        url_limit = 1
        item_limit = 50
        wiki_df = download_wikipedia(self.temp_dir, dump_date=dump_date, url_limit=url_limit, item_limit=item_limit)
        sample = wiki_df.take(10)
        assert len(sample) == 10


if __name__ == '__main__':
    unittest.main()
