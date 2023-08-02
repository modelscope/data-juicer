import unittest

from data_juicer.ops.mapper.remove_bibliography_mapper import \
    RemoveBibliographyMapper


class RemoveBibliographyMapperTest(unittest.TestCase):

    def setUp(self):
        self.op = RemoveBibliographyMapper()

    def _run_remove_bibliography(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_bibliography_case(self):

        samples = [{
            'text':
            "%%\n%% This is file `sample-sigconf.tex\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{sample-base}\n\\end{document}\n\\endinput\n%%\n%% End of file `sample-sigconf.tex'.\n",  # noqa: E501
            'target':
            '%%\n%% This is file `sample-sigconf.tex\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n'  # noqa: E501
        }]

        self._run_remove_bibliography(samples)

    def test_ref_case(self):

        samples = [{
            'text':
            "%%\n%% This is file `sample-sigconf.tex\\clearpage\n\\begin{references}\n\\end{document}\n\\endinput\n%%\n%% End of file `sample-sigconf.tex'.\n",  # noqa: E501
            'target':
            '%%\n%% This is file `sample-sigconf.tex\\clearpage\n'  # noqa: E501
        }]

        self._run_remove_bibliography(samples)


if __name__ == '__main__':
    unittest.main()
