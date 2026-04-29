#!/usr/bin/env python3
"""
Sentiment Analysis Accuracy Testbed

Tests the accuracy of sentiment classification across various categories:
- Positive examples
- Negative examples
- Neutral examples
- Mixed sentiment
- Sarcasm and irony
- Domain-specific language

Usage:
 python tests/testbed_sentiment.py [--bundle ./programs/mood]
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TestExample:
    """A single test case."""
    input: str
    expected: str
    category: str = "general"
    difficulty: str = "easy"  # easy, medium, hard


# ============================================================================
# SENTIMENT TEST DATASETS
# ============================================================================

POSITIVE_EXAMPLES = [
    TestExample("I absolutely love this product!", "positive", "explicit"),
    TestExample("This is the best thing ever!", "positive", "explicit"),
    TestExample("Fantastic experience, highly recommend!", "positive", "explicit"),
    TestExample("Amazing quality and fast shipping!", "positive", "explicit"),
    TestExample("So happy with my purchase, worth every penny!", "positive", "explicit"),
    TestExample("Excellent customer service, very helpful!", "positive", "explicit"),
    TestExample("Great value for money!", "positive", "explicit"),
    TestExample("This made my day, thank you!", "positive", "explicit"),
    TestExample("Perfect! Exactly what I needed!", "positive", "explicit"),
    TestExample("Outstanding performance and reliability!", "positive", "explicit"),
]

NEGATIVE_EXAMPLES = [
    TestExample("This is terrible, worst purchase ever!", "negative", "explicit"),
    TestExample("Complete waste of money, very disappointed!", "negative", "explicit"),
    TestExample("Horrible quality, broke after one use!", "negative", "explicit"),
    TestExample("Awful experience, never buying again!", "negative", "explicit"),
    TestExample("The product arrived damaged and customer service was unhelpful!", "negative", "explicit"),
    TestExample("So frustrated with this, total garbage!", "negative", "explicit"),
    TestExample("Do not buy! Scam!", "negative", "explicit"),
    TestExample("Regret purchasing this, complete junk!", "negative", "explicit"),
    TestExample("Worst decision ever, absolutely hate it!", "negative", "explicit"),
    TestExample("Disappointed and angry, want my money back!", "negative", "explicit"),
]

NEUTRAL_EXAMPLES = [
    TestExample("It's okay, nothing special.", "neutral", "neutral"),
    TestExample("Average product, works as expected.", "neutral", "neutral"),
    TestExample("Neither good nor bad, just mediocre.", "neutral", "neutral"),
    TestExample("Standard quality, no complaints but no praise either.", "neutral", "neutral"),
    TestExample("It works, that's about it.", "neutral", "neutral"),
    TestExample("Middle of the road, could be better could be worse.", "neutral", "neutral"),
    TestExample("Decent enough for the price.", "neutral", "neutral"),
    TestExample("Not impressed but not disappointed either.", "neutral", "neutral"),
]

MIXED_SENTIMENT = [
    TestExample("Great product but terrible shipping!", "mixed", "mixed"),
    TestExample("Love the features but hate the price!", "mixed", "mixed"),
    TestExample("Good quality but took forever to arrive!", "mixed", "mixed"),
    TestExample("Amazing customer service but product broke quickly!", "mixed", "mixed"),
    TestExample("Perfect design but way overpriced!", "mixed", "mixed"),
]

SARCASM_IRONY = [
    TestExample("Oh great, another bug! Just what I needed!", "negative", "sarcasm"),
    TestExample("Fantastic, my package is lost again!", "negative", "sarcasm"),
    TestExample("Wow, such amazing quality... not!", "negative", "sarcasm"),
    TestExample("Sure, if by 'quality' you mean 'disposable'!", "negative", "sarcasm"),
    TestExample("Best purchase ever... said no one ever!", "negative", "sarcasm"),
]

CONTEXT_DEPENDENT = [
    TestExample("This is sick!", "positive", "slang"),  # Could be positive in slang
    TestExample("That's fire!", "positive", "slang"),
    TestExample("It's lit!", "positive", "slang"),
    TestExample("This is trash!", "negative", "slang"),
    TestExample("What a banger!", "positive", "slang"),
]

DOMAIN_SPECIFIC = [
    # Tech product reviews
    TestExample("The battery life is incredible, lasts all day!", "positive", "tech"),
    TestExample("Screen resolution is disappointing for the price!", "negative", "tech"),
    TestExample("Fast processor but runs hot!", "mixed", "tech"),

    # Restaurant reviews
    TestExample("Delicious food but slow service!", "mixed", "restaurant"),
    TestExample("Best meal I've ever had!", "positive", "restaurant"),
    TestExample("Cold food and rude waiter!", "negative", "restaurant"),

    # Movie reviews
    TestExample("Masterpiece of cinema!", "positive", "movie"),
    TestExample("Worst movie of the year!", "negative", "movie"),
    TestExample("Great acting but weak plot!", "mixed", "movie"),
]

# ============================================================================
# TEST RUNNER
# ============================================================================

class SentimentTestbed:
    """Runs sentiment analysis accuracy tests."""

    def __init__(self, classifier_fn):
        self.classifier_fn = classifier_fn
        self.results: List[Dict] = []

    def run_test(self, example: TestExample) -> bool:
        """Run a single test case."""
        try:
            result = self.classifier_fn(example.input)
            # Normalize result
            result_lower = result.lower().strip()
            expected_lower = example.expected.lower().strip()

            # Check if result matches expected (exact match or contains)
            correct = (
                result_lower == expected_lower or
                expected_lower in result_lower or
                result_lower in expected_lower
            )

            self.results.append({
                'input': example.input,
                'expected': example.expected,
                'predicted': result,
                'correct': correct,
                'category': example.category,
                'difficulty': example.difficulty
            })

            return correct
        except Exception as e:
            self.results.append({
                'input': example.input,
                'expected': example.expected,
                'predicted': f"ERROR: {str(e)}",
                'correct': False,
                'category': example.category,
                'difficulty': example.difficulty,
                'error': str(e)
            })
            return False

    def run_all(self, examples: List[TestExample]) -> Dict:
        """Run all test cases and return statistics."""
        self.results.clear()
        correct = 0
        for ex in examples:
            if self.run_test(ex):
                correct += 1

        total = len(examples)
        accuracy = correct / total if total > 0 else 0

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'results': self.results
        }

    def get_category_breakdown(self) -> Dict[str, Dict]:
        """Get accuracy breakdown by category."""
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'correct': 0}
            categories[cat]['total'] += 1
            if result['correct']:
                categories[cat]['correct'] += 1

        for cat in categories:
            total = categories[cat]['total']
            correct = categories[cat]['correct']
            categories[cat]['accuracy'] = correct / total if total > 0 else 0

        return categories


def create_test_suite() -> List[TestExample]:
    """Create comprehensive test suite."""
    all_examples = []
    all_examples.extend(POSITIVE_EXAMPLES)
    all_examples.extend(NEGATIVE_EXAMPLES)
    all_examples.extend(NEUTRAL_EXAMPLES)
    all_examples.extend(MIXED_SENTIMENT)
    all_examples.extend(SARCASM_IRONY)
    all_examples.extend(CONTEXT_DEPENDENT)
    all_examples.extend(DOMAIN_SPECIFIC)
    return all_examples


def print_report(stats: Dict, category_breakdown: Dict):
    """Print detailed accuracy report."""
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS ACCURACY REPORT")
    print("="*70)

    print(f"\nOverall Performance:")
    print(f"  Total tests: {stats['total']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Accuracy: {stats['accuracy']*100:.1f}%")

    print(f"\nPerformance by Category:")
    print("-"*70)
    for cat, data in sorted(category_breakdown.items()):
        acc = data.get('accuracy', 0) * 100
        total = data['total']
        correct = data['correct']
        print(f"  {cat:20s}: {correct:3d}/{total:3d} ({acc:5.1f}%)")

    print("\n" + "="*70)

    # Show errors
    errors = [r for r in stats['results'] if not r['correct']]
    if errors:
        print(f"\nMisclassifications ({len(errors)}):")
        print("-"*70)
        for err in errors[:10]:  # Show first 10
            print(f"  Input: {err['input'][:60]}...")
            print(f"  Expected: {err['expected']}, Got: {err['predicted']}")
            print()


def main():
    # Add parent to path once for module import
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    import pawpaw

    parser = argparse.ArgumentParser(description='Sentiment Analysis Testbed')
    parser.add_argument('--bundle', type=str, default='./programs/mood',
                    help='Path to pawpaw program bundle')
    parser.add_argument('--batch', action='store_true',
                    help='Use batch_call API for testing')
    args = parser.parse_args()
    classifier = pawpaw.load(args.bundle)

    # Create test suite
    test_examples = create_test_suite()
    print(f"Running {len(test_examples)} sentiment analysis tests...")

    # Run tests
    testbed = SentimentTestbed(classifier)

    if args.batch:
        # Batch mode - process once and populate results directly
        inputs = [ex.input for ex in test_examples]
        results = classifier.batch_call(inputs)
        for ex, result in zip(test_examples, results):
            result_lower = result.lower().strip()
            expected_lower = ex.expected.lower().strip()
            correct = (
                result_lower == expected_lower or
                expected_lower in result_lower or
                result_lower in expected_lower
            )
            testbed.results.append({
                'input': ex.input,
                'expected': ex.expected,
                'predicted': result,
                'correct': correct,
                'category': ex.category,
                'difficulty': ex.difficulty
            })
        # Compute batch stats
        correct = sum(1 for r in testbed.results if r['correct'])
        total = len(testbed.results)
        stats = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'results': testbed.results
        }
    else:
        # Single call mode
        stats = testbed.run_all(test_examples)

    category_breakdown = testbed.get_category_breakdown()

    # Print report
    print_report(stats, category_breakdown)

    # Return exit code based on accuracy
    if stats['accuracy'] < 0.5:
        print("\nWarning: Accuracy below 50%")
        return 1
    elif stats['accuracy'] < 0.7:
        print("\nAccuracy is moderate (50-70%)")
        return 0
    else:
        print("\nGood accuracy (>70%)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
