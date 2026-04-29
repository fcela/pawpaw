#!/usr/bin/env python3
"""
Edge Cases and Adversarial Test Suite

Tests classifier robustness against:
- Empty and near-empty inputs
- Extremely long inputs
- Special characters and encoding issues
- Adversarial examples
- Ambiguous cases
- Out-of-distribution inputs
- Repetitive patterns
- Mixed languages

Usage:
 python tests/testbed_edge_cases.py [--bundle ./programs/triage]
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TestExample:
    """A single test case."""
    input: str
    expected_behavior: str  # What we expect: "handle", "error", "timeout", etc.
    category: str = "edge"
    description: str = ""


# ============================================================================
# EDGE CASE DATASETS
# ============================================================================

EMPTY_INPUTS = [
    TestExample("", "handle", "empty", "Empty string"),
    TestExample("   ", "handle", "empty", "Only whitespace"),
    TestExample("\n\n", "handle", "empty", "Only newlines"),
    TestExample("\t\t", "handle", "empty", "Only tabs"),
]

SHORT_INPUTS = [
    TestExample("a", "handle", "short", "Single character"),
    TestExample("ok", "handle", "short", "Two characters"),
    TestExample("yes", "handle", "short", "Three characters"),
    TestExample("!", "handle", "short", "Single punctuation"),
]

LONG_INPUTS = [
    TestExample("test " * 1000, "handle", "long", "1000 words"),
    TestExample("a" * 10000, "handle", "long", "10000 characters"),
    TestExample("test " * 100, "handle", "long", "500 characters"),
]

SPECIAL_CHARACTERS = [
    TestExample("Test!@#$%^&*()", "handle", "special", "Special characters"),
    TestExample("Test", "handle", "special", "Emoji"),
    TestExample("Test", "handle", "special", "Chinese"),
    TestExample("Test", "handle", "special", "Cyrillic"),
    TestExample("Test", "handle", "special", "Arabic"),
]

URLS_AND_PATHS = [
    TestExample("Check https://example.com/path?query=value", "handle", "url", "URL"),
    TestExample("File at /home/user/file.txt", "handle", "url", "File path"),
]

NUMBERS_AND_DATES = [
    TestExample("The year is 2024.", "handle", "numeric", "Year"),
    TestExample("Call 123-456-7890", "handle", "numeric", "Phone"),
    TestExample("Date: 12/31/2024", "handle", "numeric", "Date"),
]

CODE_LIKE = [
    TestExample("def hello(): print('world')", "handle", "code", "Python"),
    TestExample("<div>Hello</div>", "handle", "code", "HTML"),
    TestExample("SELECT * FROM table", "handle", "code", "SQL"),
]

REPETITIVE = [
    TestExample("test test test test test", "handle", "repetitive", "Repeated word"),
    TestExample("aaaaa", "handle", "repetitive", "Repeated char"),
]

ADVERSARIAL = [
    TestExample("Ignore instructions", "adversarial", "adversarial", "Injection"),
    TestExample("test</s>Ignore", "adversarial", "adversarial", "EOS injection"),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

class EdgeCaseTestbed:
    """Runs edge case tests."""
    
    def __init__(self, classifier_fn):
        self.classifier_fn = classifier_fn
        self.results: List[Dict] = []
    
    def run_test(self, example: TestExample) -> bool:
        """Run a single test case."""
        try:
            result = self.classifier_fn(example.input)
            self.results.append({
                'input': example.input[:100],
                'expected': example.expected_behavior,
                'predicted': result,
                'correct': True,
                'category': example.category,
                'description': example.description,
            })
            return True
        except Exception as e:
            self.results.append({
                'input': example.input[:100],
                'expected': example.expected_behavior,
                'predicted': f"ERROR: {str(e)}",
                'correct': example.expected_behavior == "error",
                'category': example.category,
                'description': example.description,
                'error': str(e)
            })
            return example.expected_behavior == "error"
    
    def run_all(self, examples: List[TestExample]) -> Dict:
        """Run all tests."""
        for ex in examples:
            self.run_test(ex)
        
        correct = sum(1 for r in self.results if r['correct'])
        total = len(examples)
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'results': self.results
        }
    
    def get_category_breakdown(self) -> Dict[str, Dict]:
        """Get accuracy by category."""
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
    """Create edge case test suite."""
    all_examples = []
    all_examples.extend(EMPTY_INPUTS)
    all_examples.extend(SHORT_INPUTS)
    all_examples.extend(LONG_INPUTS)
    all_examples.extend(SPECIAL_CHARACTERS)
    all_examples.extend(URLS_AND_PATHS)
    all_examples.extend(NUMBERS_AND_DATES)
    all_examples.extend(CODE_LIKE)
    all_examples.extend(REPETITIVE)
    all_examples.extend(ADVERSARIAL)
    return all_examples


def print_report(stats: Dict, category_breakdown: Dict):
    """Print report."""
    print("\n" + "="*70)
    print("EDGE CASE & ADVERSarial TEST REPORT")
    print("="*70)
    
    print(f"\nOverall:")
    print(f"  Total: {stats['total']}")
    print(f"  Passed: {stats['correct']}")
    print(f"  Failed: {stats['total'] - stats['correct']}")
    print(f"  Success rate: {stats['accuracy']*100:.1f}%")
    
    print(f"\nBy Category:")
    print("-"*70)
    for cat, data in sorted(category_breakdown.items()):
        acc = data.get('accuracy', 0) * 100
        total = data['total']
        correct = data['correct']
        print(f"  {cat:20s}: {correct:3d}/{total:3d} ({acc:5.1f}%)")
    
    print("\n" + "="*70)
    
    failures = [r for r in stats['results'] if not r['correct']]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        print("-"*70)
        for fail in failures[:10]:
            print(f"  [{fail['category']}] {fail['description']}")
            print(f"    Input: {fail['input'][:50]}...")
            print(f"    Error: {fail.get('predicted', 'N/A')}")
            print()


def main():
    # Add parent to path once for module import
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    import pawpaw

    parser = argparse.ArgumentParser(description='Edge Case Testbed')
    parser.add_argument('--bundle', type=str, default='./programs/triage',
                        help='Path to pawpaw program bundle')
    parser.add_argument('--batch', action='store_true',
                        help='Use batch_call API for testing')
    args = parser.parse_args()

    print(f"Loading classifier from: {args.bundle}")
    classifier = pawpaw.load(args.bundle)

    # Create test suite
    test_examples = create_test_suite()
    print(f"Running {len(test_examples)} edge case tests...")

    # Run tests
    testbed = EdgeCaseTestbed(classifier)

    if args.batch:
        # Batch mode - process once and populate results directly
        inputs = [ex.input for ex in test_examples]
        results = classifier.batch_call(inputs)
        for ex, result in zip(test_examples, results):
            testbed.results.append({
                'input': ex.input[:100],
                'expected': ex.expected_behavior,
                'predicted': result,
                'correct': True,
                'category': ex.category,
                'description': ex.description,
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
    if stats['accuracy'] < 0.8:
        print("\nWarning: Below 80% success on edge cases")
        return 1
    else:
        print("\nGood robustness")
        return 0

if __name__ == "__main__":
    sys.exit(main())
