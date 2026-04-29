#!/usr/bin/env python3
"""
Intent Classification Accuracy Testbed

Tests the accuracy of intent recognition across various categories:
- Questions (what, when, where, why, how)
- Requests (action, information, help)
- Commands (imperative)
- Statements (declarative)
- Greetings and farewells
- Confirmations and denials

Usage:
 python tests/testbed_intent.py [--bundle ./programs/triage]
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
    expected: str
    category: str = "general"


# ============================================================================
# INTENT TEST DATASETS
# ============================================================================

QUESTIONS_WHAT = [
    TestExample("What is the weather today?", "question", "what"),
    TestExample("What time is it?", "question", "what"),
    TestExample("What are your hours?", "question", "what"),
    TestExample("What does this button do?", "question", "what"),
    TestExample("What's the meaning of life?", "question", "what"),
]

QUESTIONS_WHEN = [
    TestExample("When does the store open?", "question", "when"),
    TestExample("When is the deadline?", "question", "when"),
    TestExample("When will it be ready?", "question", "when"),
    TestExample("When did this happen?", "question", "when"),
]

QUESTIONS_WHERE = [
    TestExample("Where is the nearest station?", "question", "where"),
    TestExample("Where can I find this?", "question", "where"),
    TestExample("Where are you located?", "question", "where"),
]

QUESTIONS_WHY = [
    TestExample("Why is this happening?", "question", "why"),
    TestExample("Why did you do that?", "question", "why"),
    TestExample("Why is the sky blue?", "question", "why"),
]

QUESTIONS_HOW = [
    TestExample("How do I reset my password?", "question", "how"),
    TestExample("How much does this cost?", "question", "how"),
    TestExample("How far is it?", "question", "how"),
    TestExample("How can I help you?", "question", "how"),
]

REQUESTS_ACTION = [
    TestExample("Please help me with this issue.", "request", "action"),
    TestExample("Can you send me the document?", "request", "action"),
    TestExample("I need assistance with my account.", "request", "action"),
    TestExample("Could you explain this to me?", "request", "action"),
    TestExample("Would you mind checking that?", "request", "action"),
]

REQUESTS_INFORMATION = [
    TestExample("Tell me more about this product.", "request", "information"),
    TestExample("I'd like to know the details.", "request", "information"),
    TestExample("Can you provide more context?", "request", "information"),
]

COMMANDS = [
    TestExample("Open the file now!", "command", "imperative"),
    TestExample("Delete all temporary files.", "command", "imperative"),
    TestExample("Send this to everyone.", "command", "imperative"),
    TestExample("Stop the process immediately!", "command", "imperative"),
    TestExample("Run the diagnostics.", "command", "imperative"),
]

STATEMENTS = [
    TestExample("I think this is working.", "statement", "declarative"),
    TestExample("The weather is nice today.", "statement", "declarative"),
    TestExample("I have a meeting at 3pm.", "statement", "declarative"),
    TestExample("This is my first time here.", "statement", "declarative"),
]

GREETINGS = [
    TestExample("Hello!", "greeting", "greeting"),
    TestExample("Hi there, how are you?", "greeting", "greeting"),
    TestExample("Good morning!", "greeting", "greeting"),
    TestExample("Hey, what's up?", "greeting", "greeting"),
]

FAREWELLS = [
    TestExample("Goodbye!", "farewell", "farewell"),
    TestExample("See you later!", "farewell", "farewell"),
    TestExample("Have a nice day!", "farewell", "farewell"),
    TestExample("Take care!", "farewell", "farewell"),
]

CONFIRMATIONS = [
    TestExample("Yes, that's correct.", "confirmation", "yes"),
    TestExample("Absolutely, I agree.", "confirmation", "yes"),
    TestExample("Sure, go ahead.", "confirmation", "yes"),
    TestExample("Definitely!", "confirmation", "yes"),
]

DENIALS = [
    TestExample("No, that's not right.", "denial", "no"),
    TestExample("I disagree with that.", "denial", "no"),
    TestExample("Absolutely not!", "denial", "no"),
    TestExample("No way!", "denial", "no"),
]

AMBIGUOUS = [
    TestExample("Okay.", "ambiguous", "ambiguous"),
    TestExample("Maybe.", "ambiguous", "ambiguous"),
    TestExample("I'm not sure.", "ambiguous", "ambiguous"),
    TestExample("We'll see.", "ambiguous", "ambiguous"),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

class IntentTestbed:
    """Runs intent classification accuracy tests."""
    
    def __init__(self, classifier_fn):
        self.classifier_fn = classifier_fn
        self.results: List[Dict] = []
    
    def run_test(self, example: TestExample) -> bool:
        """Run a single test case."""
        try:
            result = self.classifier_fn(example.input)
            result_lower = result.lower().strip()
            expected_lower = example.expected.lower().strip()
            
            # Check if result matches expected
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
            })
            return correct
        except Exception as e:
            self.results.append({
                'input': example.input,
                'expected': example.expected,
                'predicted': f"ERROR: {str(e)}",
                'correct': False,
                'category': example.category,
                'error': str(e)
            })
            return False
    
    def run_all(self, examples: List[TestExample]) -> Dict:
        """Run all test cases."""
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
    """Create comprehensive intent test suite."""
    all_examples = []
    all_examples.extend(QUESTIONS_WHAT)
    all_examples.extend(QUESTIONS_WHEN)
    all_examples.extend(QUESTIONS_WHERE)
    all_examples.extend(QUESTIONS_WHY)
    all_examples.extend(QUESTIONS_HOW)
    all_examples.extend(REQUESTS_ACTION)
    all_examples.extend(REQUESTS_INFORMATION)
    all_examples.extend(COMMANDS)
    all_examples.extend(STATEMENTS)
    all_examples.extend(GREETINGS)
    all_examples.extend(FAREWELLS)
    all_examples.extend(CONFIRMATIONS)
    all_examples.extend(DENIALS)
    all_examples.extend(AMBIGUOUS)
    return all_examples


def print_report(stats: Dict, category_breakdown: Dict):
    """Print accuracy report."""
    print("\n" + "="*70)
    print("INTENT CLASSIFICATION ACCURACY REPORT")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Total tests: {stats['total']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Accuracy: {stats['accuracy']*100:.1f}%")
    
    print(f"\nPerformance by Intent Category:")
    print("-"*70)
    for cat, data in sorted(category_breakdown.items()):
        acc = data.get('accuracy', 0) * 100
        total = data['total']
        correct = data['correct']
        print(f"  {cat:20s}: {correct:3d}/{total:3d} ({acc:5.1f}%)")
    
    print("\n" + "="*70)
    
    errors = [r for r in stats['results'] if not r['correct']]
    if errors:
        print(f"\nMisclassifications ({len(errors)}):")
        print("-"*70)
        for err in errors[:10]:
            print(f"  Input: {err['input'][:60]}...")
            print(f"    Expected: {err['expected']}, Got: {err['predicted']}")
            print()


def main():
    # Add parent to path once for module import
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    import pawpaw

    parser = argparse.ArgumentParser(description='Intent Classification Testbed')
    parser.add_argument('--bundle', type=str, default='./programs/triage',
                        help='Path to pawpaw program bundle')
    parser.add_argument('--batch', action='store_true',
                        help='Use batch_call API')
    args = parser.parse_args()

    print(f"Loading classifier from: {args.bundle}")
    classifier = pawpaw.load(args.bundle)

    test_examples = create_test_suite()
    print(f"Running {len(test_examples)} intent classification tests...")

    testbed = IntentTestbed(classifier)

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
    print_report(stats, category_breakdown)
    if stats['accuracy'] < 0.5:
        print("\n⚠️  Warning: Accuracy below 50%")
        return 1
    elif stats['accuracy'] < 0.7:
        print("\n⚠️  Accuracy is moderate (50-70%)")
        return 0
    else:
        print("\n✅ Good accuracy (>70%)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
