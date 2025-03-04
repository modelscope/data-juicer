import random
import re
import string
from typing import Callable, Dict


class TextModifier:
    """Advanced text modification strategies for creating near-duplicates"""

    @staticmethod
    def character_swap(text: str, rate: float = 0.05) -> str:
        """Swap characters randomly"""
        chars = list(text)
        swaps = max(1, int(len(chars) * rate))

        for _ in range(swaps):
            if len(chars) < 2:
                break
            i = random.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

        return ''.join(chars)

    @staticmethod
    def word_replacement(text: str, rate: float = 0.1) -> str:
        """Replace words with similar ones"""
        words = text.split()
        replacements = max(1, int(len(words) * rate))

        for _ in range(replacements):
            if not words:
                break
            i = random.randint(0, len(words) - 1)

            # Simple replacement strategy (could be enhanced with word
            # embeddings)
            if len(words[i]) > 3:
                # Replace with a slightly modified version
                chars = list(words[i])
                pos = random.randint(0, len(chars) - 1)
                chars[pos] = random.choice(string.ascii_lowercase)
                words[i] = ''.join(chars)

        return ' '.join(words)

    @staticmethod
    def sentence_reordering(text: str, rate: float = 0.3) -> str:
        """Reorder sentences within paragraphs"""
        paragraphs = text.split('\n')

        for i, paragraph in enumerate(paragraphs):
            # Split into sentences (simple approach)
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            if len(sentences) > 1:
                # Shuffle some sentences
                num_to_shuffle = max(1, int(len(sentences) * rate))
                indices = random.sample(range(len(sentences)),
                                        k=num_to_shuffle)

                # Extract sentences to shuffle
                to_shuffle = [sentences[j] for j in sorted(indices)]
                random.shuffle(to_shuffle)

                # Put back in original positions
                for idx, j in enumerate(sorted(indices)):
                    sentences[j] = to_shuffle[idx]

                paragraphs[i] = ' '.join(sentences)

        return '\n'.join(paragraphs)

    @staticmethod
    def html_modification(text: str, rate: float = 0.2) -> str:
        """Modify HTML attributes while preserving structure"""

        # This is a simplified version - a real implementation would
        # use an HTML parser

        # Modify attributes in tags
        def replace_attr(match):
            tag = match.group(1)
            attrs = match.group(2)

            # Randomly modify some attributes
            if random.random() < rate:
                # Add a random attribute
                attrs += f' data-random="{random.randint(1, 1000)}"'

            return f'<{tag} {attrs}>'

        modified = re.sub(r'<(\w+)\s+([^>]+)>', replace_attr, text)
        return modified

    @staticmethod
    def whitespace_modification(text: str) -> str:
        """Modify whitespace without changing content"""
        # Replace multiple spaces with single space
        modified = re.sub(r'\s+', ' ', text)

        # Randomly add extra newlines
        sentences = re.split(r'(?<=[.!?])\s+', modified)
        for i in range(len(sentences) - 1):
            if random.random() < 0.2:
                sentences[i] = sentences[i] + '\n'

        return ' '.join(sentences)

    @staticmethod
    def case_modification(text: str, rate: float = 0.1) -> str:
        """Change case of some words"""
        words = text.split()
        modifications = max(1, int(len(words) * rate))

        for _ in range(modifications):
            if not words:
                break
            i = random.randint(0, len(words) - 1)

            # Skip very short words
            if len(words[i]) < 3:
                continue

            # Apply case modification
            mod_type = random.choice(['upper', 'lower', 'title'])
            if mod_type == 'upper':
                words[i] = words[i].upper()
            elif mod_type == 'lower':
                words[i] = words[i].lower()
            else:
                words[i] = words[i].title()

        return ' '.join(words)


class ModificationStrategy:
    """Combines multiple modification strategies with configurable weights"""

    def __init__(self, strategies: Dict[Callable, float] = None):
        """Initialize with strategies and their weights

        Args:
            strategies: Dictionary mapping strategy functions to their weights
        """
        if strategies is None:
            # Default strategies and weights
            self.strategies = {
                TextModifier.character_swap: 0.2,
                TextModifier.word_replacement: 0.3,
                TextModifier.sentence_reordering: 0.2,
                TextModifier.whitespace_modification: 0.1,
                TextModifier.case_modification: 0.2
            }
        else:
            self.strategies = strategies

    def apply(self, text: str, intensity: float = 0.5) -> str:
        """Apply modification strategies based on weights and intensity

        Args:
            text: Text to modify
            intensity: Overall modification intensity (0.0 to 1.0)

        Returns:
            Modified text
        """
        modified = text

        # Normalize weights
        total_weight = sum(self.strategies.values())
        normalized_weights = {
            k: v / total_weight
            for k, v in self.strategies.items()
        }

        # Apply strategies based on weights and intensity
        for strategy, weight in normalized_weights.items():
            if random.random() < weight * intensity:
                modified = strategy(modified)

        return modified
