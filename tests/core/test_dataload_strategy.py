import unittest
from data_juicer.core.data.load_strategy import (
    DataLoadStrategyRegistry, DataLoadStrategy, StrategyKey
)
from data_juicer.core.executor.base import ExecutorType

class MockStrategy(DataLoadStrategy):
    def load_data(self):
        pass

class DataLoadStrategyRegistryTest(unittest.TestCase):
    def setUp(self):
        # Clear existing strategies before each test
        DataLoadStrategyRegistry._strategies = {}
    
    def test_exact_match(self):
        # Register a specific strategy
        @DataLoadStrategyRegistry.register(ExecutorType.LOCAL, 'ondisk', 'json')
        class TestStrategy(MockStrategy):
            pass

        # Test exact match
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'json')
        self.assertEqual(strategy, TestStrategy)

        # Test no match
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'csv')
        self.assertIsNone(strategy)

    def test_wildcard_matching(self):
        # Register strategies with different wildcard patterns
        @DataLoadStrategyRegistry.register(ExecutorType.LOCAL, 'ondisk', '*')
        class AllFilesStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(ExecutorType.LOCAL, '*', '*')
        class AllLocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(ExecutorType.ANY, '*', '*')
        class FallbackStrategy(MockStrategy):
            pass

        # Test specific matches
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'json')
        self.assertEqual(strategy, AllFilesStrategy)  # Should match most specific wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'remote', 'json')
        self.assertEqual(strategy, AllLocalStrategy)  # Should match second level wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.RAY, 'remote', 'json')
        self.assertEqual(strategy, FallbackStrategy)  # Should match most general wildcard

    def test_specificity_priority(self):
        @DataLoadStrategyRegistry.register(ExecutorType.ANY, '*', '*')
        class GeneralStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(ExecutorType.LOCAL, '*', '*')
        class LocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(ExecutorType.LOCAL, 'ondisk', '*')
        class LocalOndiskStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(ExecutorType.LOCAL, 'ondisk', 'json')
        class ExactStrategy(MockStrategy):
            pass

        # Test matching priority
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'json')
        self.assertEqual(strategy, ExactStrategy)  # Should match exact first

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'csv')
        self.assertEqual(strategy, LocalOndiskStrategy)  # Should match one wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'remote', 'json')
        self.assertEqual(strategy, LocalStrategy)  # Should match two wildcards

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.RAY, 'remote', 'json')
        self.assertEqual(strategy, GeneralStrategy)  # Should match general wildcard

    def test_pattern_matching(self):
        @DataLoadStrategyRegistry.register(
            ExecutorType.LOCAL, 'ondisk', '*.json')
        class JsonStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(
            ExecutorType.LOCAL, 'ondisk', 'data_[0-9]*')
        class NumberedDataStrategy(MockStrategy):
            pass

        # Test pattern matching
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'test.json')
        self.assertEqual(strategy, JsonStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'data_123')
        self.assertEqual(strategy, NumberedDataStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            ExecutorType.LOCAL, 'ondisk', 'test.csv')
        self.assertIsNone(strategy)

    def test_strategy_key_matches(self):
        # Test StrategyKey matching directly
        wildcard_key = StrategyKey(ExecutorType.ANY, 'ondisk', '*.json')
        specific_key = StrategyKey(ExecutorType.LOCAL, 'ondisk', 'test.json')
        
        # Exact keys don't match wildcards
        self.assertTrue(wildcard_key.matches(specific_key))
        self.assertFalse(specific_key.matches(wildcard_key))  

        # Test pattern matching
        pattern_key = StrategyKey(ExecutorType.LOCAL, '*', 'data_[0-9]*')
        match_key = StrategyKey(ExecutorType.LOCAL, 'ondisk', 'data_123')
        no_match_key = StrategyKey(ExecutorType.LOCAL, 'ondisk', 'data_abc')
        
        self.assertTrue(pattern_key.matches(match_key))
        self.assertFalse(pattern_key.matches(no_match_key))
