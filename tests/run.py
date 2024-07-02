# The code is from adapted from
# https://github.com/alibaba/FederatedScope/blob/master/tests/run.py

# Data-Juicer adopts Apache 2.0 license, the original license of this file
# is as follows:
# --------------------------------------------------------
# Copyright (c) Alibaba, Inc. and its affiliates

import argparse
import os
import sys
import unittest

from loguru import logger

from data_juicer.utils.unittest_utils import SKIPPED_TESTS

file_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(file_dir)

parser = argparse.ArgumentParser('test runner')
parser.add_argument('--tag', choices=["standalone", "ray"],
                    default="standalone",
                    help="the tag of tests being run")
parser.add_argument('--pattern', default='test_*.py', help='test file pattern')
parser.add_argument('--test_dir',
                    default='tests',
                    help='directory to be tested')
args = parser.parse_args()


class TaggedTestLoader(unittest.TestLoader):
    def __init__(self, tag="standalone"):
        super().__init__()
        self.tag = tag
    
    def loadTestsFromTestCase(self, testCaseClass):
        # set tag to testcase class
        setattr(testCaseClass, 'current_tag', self.tag)
        test_names = self.getTestCaseNames(testCaseClass)
        loaded_suite = self.suiteClass()
        for test_name in test_names:
            test_case = testCaseClass(test_name)
            test_method = getattr(test_case, test_name)
            if self.tag in getattr(test_method, '__test_tags__', ["standalone"]):
                loaded_suite.addTest(test_case)
        return loaded_suite

def gather_test_cases(test_dir, pattern, tag):
    test_to_run = unittest.TestSuite()
    test_loader = TaggedTestLoader(tag)
    discover = test_loader.discover(test_dir, pattern=pattern, top_level_dir=None)
    print(f'These tests will be skipped due to some reasons: '
          f'{SKIPPED_TESTS.modules}')
    for suite_discovered in discover:
        for test_suite in suite_discovered:
            for test_case in test_suite:
                if type(test_case) in SKIPPED_TESTS.modules.values():
                    continue
                logger.info(f'Add test case [{test_case._testMethodName}]'
                            f' from {test_case.__class__.__name__}')
                test_to_run.addTest(test_case)
    return test_to_run


def main():
    runner = unittest.TextTestRunner()
    test_suite = gather_test_cases(os.path.abspath(args.test_dir),
                                   args.pattern, args.tag)
    res = runner.run(test_suite)
    if not res.wasSuccessful():
        exit(1)


if __name__ == '__main__':
    main()
