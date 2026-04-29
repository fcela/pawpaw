#!/usr/bin/env python3
"""
Topic Classification Accuracy Testbed

Tests the accuracy of topic categorization across various domains:
- Technology
- Sports
- Politics
- Entertainment
- Business
- Science
- Health
- Travel
- Food
- Weather

Usage:
 python tests/testbed_topic.py [--bundle ./programs/domain]
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
# TOPIC TEST DATASETS
# ============================================================================

TECHNOLOGY = [
    TestExample("The new iPhone features an A17 chip with improved performance.", "technology", "tech"),
    TestExample("Python 3.12 introduces pattern matching enhancements.", "technology", "tech"),
    TestExample("Quantum computing breakthrough announced by researchers.", "technology", "tech"),
    TestExample("The software update fixes critical security vulnerabilities.", "technology", "tech"),
    TestExample("AI models are becoming more efficient at natural language tasks.", "technology", "tech"),
]

SPORTS = [
    TestExample("The team won the championship after overtime.", "sports", "sports"),
    TestExample("The athlete broke the world record in the 100m dash.", "sports", "sports"),
    TestExample("Football season starts next week with exciting matchups.", "sports", "sports"),
    TestExample("The tennis player advanced to the semifinals.", "sports", "sports"),
]

POLITICS = [
    TestExample("The president announced new economic policies today.", "politics", "politics"),
    TestExample("Voters will head to the polls next Tuesday.", "politics", "politics"),
    TestExample("The senate passed the bill with a narrow margin.", "politics", "politics"),
    TestExample("International relations tense after diplomatic incident.", "politics", "politics"),
]

ENTERTAINMENT = [
    TestExample("The movie premiered at Cannes Film Festival.", "entertainment", "entertainment"),
    TestExample("The concert sold out in minutes.", "entertainment", "entertainment"),
    TestExample("The TV series finale drew millions of viewers.", "entertainment", "entertainment"),
    TestExample("The album topped the charts this week.", "entertainment", "entertainment"),
]

BUSINESS = [
    TestExample("Stock markets rallied on positive economic data.", "business", "business"),
    TestExample("The company announced record quarterly earnings.", "business", "business"),
    TestExample("Merger talks between the two corporations ongoing.", "business", "business"),
    TestExample("Unemployment rate dropped to historic lows.", "business", "business"),
]

SCIENCE = [
    TestExample("Scientists discovered a new species in the Amazon.", "science", "science"),
    TestExample("The research paper was published in Nature journal.", "science", "science"),
    TestExample("Climate study reveals alarming trends.", "science", "science"),
    TestExample("Physicists confirm existence of new particle.", "science", "science"),
]

HEALTH = [
    TestExample("New vaccine shows promising results in clinical trials.", "health", "health"),
    TestExample("Exercise and diet are key to preventing heart disease.", "health", "health"),
    TestExample("Mental health awareness is increasing globally.", "health", "health"),
    TestExample("The hospital implemented new safety protocols.", "health", "health"),
]

TRAVEL = [
    TestExample("The beach resort offers all-inclusive packages.", "travel", "travel"),
    TestExample("Flight delays affected thousands of passengers.", "travel", "travel"),
    TestExample("Tourism industry recovering after pandemic.", "travel", "travel"),
    TestExample("The city's historic district is a must-see.", "travel", "travel"),
]

FOOD = [
    TestExample("The restaurant received a Michelin star.", "food", "food"),
    TestExample("This recipe uses traditional Italian ingredients.", "food", "food"),
    TestExample("The food festival attracts chefs from around the world.", "food", "food"),
    TestExample("Organic produce is gaining popularity.", "food", "food"),
]

WEATHER = [
    TestExample("Heavy rain expected throughout the week.", "weather", "weather"),
    TestExample("Temperatures will reach record highs this weekend.", "weather", "weather"),
    TestExample("A storm system is moving across the region.", "weather", "weather"),
    TestExample("Snow forecast for the mountain areas.", "weather", "weather"),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

class TopicTestbed:
    """Runs topic classification accuracy tests."""
    
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
    """Create comprehensive topic test suite."""
    all_examples = []
    all_examples.extend(TECHNOLOGY)
    all_examples.extend(SPORTS)
    all_examples.extend(POLITICS)
    all_examples.extend(ENTERTAINMENT)
    all_examples.extend(BUSINESS)
    all_examples.extend(SCIENCE)
    all_examples.extend(HEALTH)
    all_examples.extend(TRAVEL)
    all_examples.extend(FOOD)
    all_examples.extend(WEATHER)
    return all_examples


def print_report(stats: Dict, category_breakdown: Dict):
    """Print accuracy report."""
    print("\n" + "="*70)
    print("TOPIC CLASSIFICATION ACCURACY REPORT")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Total tests: {stats['total']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Accuracy: {stats['accuracy']*100:.1f}%")
    
    print(f"\nPerformance by Topic:")
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

    parser = argparse.ArgumentParser(description='Topic Classification Testbed')
    parser.add_argument('--bundle', type=str, default='./programs/domain',
                        help='Path to pawpaw program bundle')
    parser.add_argument('--batch', action='store_true',
                        help='Use batch_call API')
    args = parser.parse_args()

    print(f"Loading classifier from: {args.bundle}")
    classifier = pawpaw.load(args.bundle)

    test_examples = create_test_suite()
    print(f"Running {len(test_examples)} topic classification tests...")

    testbed = TopicTestbed(classifier)

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
