import unittest

from data_juicer.utils.auto_install_utils import _is_module_installed, _is_package_installed

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class IsXXXInstalledFuncsTest(DataJuicerTestCaseBase):

    def test_is_module_installed(self):
        self.assertTrue(_is_module_installed('datasets'))
        self.assertTrue(_is_module_installed('simhash'))

        self.assertFalse(_is_module_installed('non_existent_module'))

    def test_is_package_installed(self):
        self.assertTrue(_is_package_installed('datasets'))
        self.assertTrue(_is_package_installed('ram@git+https://github.com/xinyu1205/recognize-anything.git'))
        self.assertTrue(_is_package_installed('scenedetect[opencv]'))

        self.assertFalse(_is_package_installed('non_existent_package'))


if __name__ == '__main__':
    unittest.main()
