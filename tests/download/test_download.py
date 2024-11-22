import unittest
import tempfile
from data_juicer.download.wikipedia import (
    get_wikipedia_urls, download_wikipedia
)

class TestDownload:

    def test_wikipedia_urls(self):
        dump_date = "20241101"
        urls = get_wikipedia_urls(dump_date=dump_date)
        assert len(urls) > 3
        assert urls[0] == "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2"
        assert urls[1] == "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream2.xml-p41243p151573.bz2"
        assert urls[2] == "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream3.xml-p151574p311329.bz2"

    def test_wikipedia_download(self):
        dump_date = "20241101"
        output_directory = tempfile.gettempdir() + "/dj_temp/"
        url_limit = 5
        item_limit = 10
        wiki_df = download_wikipedia(output_directory, dump_date=dump_date, url_limit=url_limit, item_limit=item_limit)
        sample = wiki_df.take(50)
        assert len(sample) == 50


if __name__ == '__main__':
    unittest.main()
