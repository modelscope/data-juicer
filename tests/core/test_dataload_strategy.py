import unittest
from data_juicer.core.data.load_strategy import (
    DataLoadStrategyRegistry, DataLoadStrategy, StrategyKey,
    DefaultLocalDataLoadStrategy
)
from argparse import Namespace
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
class MockStrategy(DataLoadStrategy):
    def load_data(self):
        pass

class DataLoadStrategyRegistryTest(DataJuicerTestCaseBase):

    def test_exact_match(self):
        # Register a specific strategy
        DataLoadStrategyRegistry._strategies = {}
        @DataLoadStrategyRegistry.register("default", 'local', 'json')
        class TestStrategy(MockStrategy):
            pass

        # Test exact match
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'json')
        self.assertEqual(strategy, TestStrategy)

        # Test no match
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'csv')
        self.assertIsNone(strategy)

    def test_wildcard_matching(self):
        # Register strategies with different wildcard patterns
        DataLoadStrategyRegistry._strategies = {}
        @DataLoadStrategyRegistry.register("default", 'local', '*')
        class AllFilesStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", '*', '*')
        class AllLocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("*", '*', '*')
        class FallbackStrategy(MockStrategy):
            pass

        # Test specific matches
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'json')
        self.assertEqual(strategy, AllFilesStrategy)  # Should match most specific wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'remote', 'json')
        self.assertEqual(strategy, AllLocalStrategy)  # Should match second level wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "ray", 'remote', 'json')
        self.assertEqual(strategy, FallbackStrategy)  # Should match most general wildcard

    def test_specificity_priority(self):
        DataLoadStrategyRegistry._strategies = {}

        @DataLoadStrategyRegistry.register("*", '*', '*')
        class GeneralStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", '*', '*')
        class LocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", 'local', '*')
        class LocalOndiskStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", 'local', 'json')
        class ExactStrategy(MockStrategy):
            pass

        # Test matching priority
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'json')
        self.assertEqual(strategy, ExactStrategy)  # Should match exact first

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'csv')
        self.assertEqual(strategy, LocalOndiskStrategy)  # Should match one wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'remote', 'json')
        self.assertEqual(strategy, LocalStrategy)  # Should match two wildcards

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "ray", 'remote', 'json')
        self.assertEqual(strategy, GeneralStrategy)  # Should match general wildcard

    def test_pattern_matching(self):
        @DataLoadStrategyRegistry.register(
            "default", 'local', '*.json')
        class JsonStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(
            "default", 'local', 'data_[0-9]*')
        class NumberedDataStrategy(MockStrategy):
            pass

        # Test pattern matching
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'test.json')
        self.assertEqual(strategy, JsonStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'data_123')
        self.assertEqual(strategy, NumberedDataStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'test.csv')
        self.assertIsNone(strategy)

    def test_strategy_key_matches(self):
        DataLoadStrategyRegistry._strategies = {}

        # Test StrategyKey matching directly
        wildcard_key = StrategyKey("*", 'local', '*.json')
        specific_key = StrategyKey("default", 'local', 'test.json')
        
        # Exact keys don't match wildcards
        self.assertTrue(wildcard_key.matches(specific_key))
        self.assertFalse(specific_key.matches(wildcard_key))  

        # Test pattern matching
        pattern_key = StrategyKey("default", '*', 'data_[0-9]*')
        match_key = StrategyKey("default", 'local', 'data_123')
        no_match_key = StrategyKey("default", 'local', 'data_abc')
        
        self.assertTrue(pattern_key.matches(match_key))
        self.assertFalse(pattern_key.matches(no_match_key))

    def test_load_strategy_default_config(self):
        """Test load strategy with minimal config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create minimal config
        minimal_cfg = Namespace(
            path='test/path'
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, minimal_cfg)
        
        # Verify defaults are used
        assert getattr(strategy.cfg, 'text_keys', ['text']) == ['text']
        assert getattr(strategy.cfg, 'suffixes', None) is None
        assert getattr(strategy.cfg, 'add_suffix', False) is False

    def test_load_strategy_full_config(self):
        """Test load strategy with full config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with all options
        full_cfg = Namespace(
            path='test/path',
            text_keys=['content', 'title'],
            suffixes=['.txt', '.md'],
            add_suffix=True
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, full_cfg)
        
        # Verify all config values are used
        assert strategy.cfg.text_keys == ['content', 'title']
        assert strategy.cfg.suffixes == ['.txt', '.md']
        assert strategy.cfg.add_suffix is True

    def test_load_strategy_partial_config(self):
        """Test load strategy with partial config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with some options
        partial_cfg = Namespace(
            path='test/path',
            text_keys=['content'],
            # suffixes and add_suffix omitted
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, partial_cfg)
        
        # Verify mix of specified and default values
        assert strategy.cfg.text_keys == ['content']
        assert getattr(strategy.cfg, 'suffixes', None) is None
        assert getattr(strategy.cfg, 'add_suffix', False) is False

    def test_load_strategy_empty_config(self):
        """Test load strategy with empty config"""
        DataLoadStrategyRegistry._strategies = {}
        
        # Create empty config
        empty_cfg = Namespace()
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, empty_cfg)
        
        # Verify all defaults are used
        assert getattr(strategy.cfg, 'text_keys', ['text']) == ['text']
        assert getattr(strategy.cfg, 'suffixes', None) is None
        assert getattr(strategy.cfg, 'add_suffix', False) is False
        


if __name__ == '__main__':
    unittest.main()