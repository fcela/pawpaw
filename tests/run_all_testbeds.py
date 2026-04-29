#!/usr/bin/env python3
"""
Master test runner for all classification testbeds.

Runs all accuracy test suites and generates a comprehensive report.

Usage:
    python tests/run_all_testbeds.py [--bundle ./programs/triage]
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TestbedResult:
    """Result from running a testbed."""
    name: str
    accuracy: float
    total_tests: int
    passed: int
    exit_code: int
    output: str


def run_testbed(testbed_path: str, bundle: str, use_batch: bool = False) -> TestbedResult:
    """Run a single testbed and capture results."""
    cmd = [sys.executable, testbed_path, '--bundle', bundle]
    if use_batch:
        cmd.append('--batch')
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse output for accuracy
        accuracy = 0.0
        total = 0
        passed = 0
        
        for line in result.stdout.split('\n'):
            if 'Accuracy:' in line:
            try:
                        accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    except (ValueError, IndexError):
                        pass
                    elif 'Total tests:' in line:
                        try:
                            total = int(line.split(':')[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif 'Correct:' in line:
                        try:
                            passed = int(line.split(':')[1].strip())
                        except (ValueError, IndexError):
                            pass
        
        return TestbedResult(
            name=Path(testbed_path).stem.replace('testbed_', ''),
            accuracy=accuracy,
            total_tests=total,
            passed=passed,
            exit_code=result.returncode,
            output=result.stdout
        )
    except subprocess.TimeoutExpired:
        return TestbedResult(
            name=Path(testbed_path).stem.replace('testbed_', ''),
            accuracy=0.0,
            total_tests=0,
            passed=0,
            exit_code=-1,
            output="TIMEOUT"
        )
    except Exception as e:
        return TestbedResult(
            name=Path(testbed_path).stem.replace('testbed_', ''),
            accuracy=0.0,
            total_tests=0,
            passed=0,
            exit_code=-2,
            output=f"ERROR: {str(e)}"
        )


def print_summary(results: List[TestbedResult]):
    """Print summary report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ACCURACY TEST REPORT")
    print("="*80)
    
    print(f"\n{'Testbed':<30} {'Accuracy':<12} {'Passed':<10} {'Total':<10} {'Status'}")
    print("-"*80)
    
    total_accuracy = 0.0
    total_passed = 0
    total_tests = 0
    
    for result in results:
        status = "✅ PASS" if result.exit_code == 0 else "❌ FAIL"
        print(f"{result.name:<30} {result.accuracy*100:>6.1f}%   {result.passed:>6}     {result.total_tests:>6}   {status}")
        total_accuracy += result.accuracy
        total_passed += result.passed
        total_tests += result.total_tests
    
    print("-"*80)
    
    if results:
        avg_accuracy = total_accuracy / len(results) * 100
    else:
        avg_accuracy = 0.0
        print(f"\nOverall: {total_passed}/{total_tests} tests passed")
        print(f"Average accuracy: {avg_accuracy:.1f}%")
        
        if avg_accuracy >= 80:
            print("\n✅ All testbeds passed with good accuracy!")
        elif avg_accuracy >= 60:
            print("\n⚠️  Moderate accuracy - review failed tests")
        else:
            print("\n❌ Poor accuracy - classifier needs improvement")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Run all classification testbeds')
    parser.add_argument('--bundle', type=str, default='./programs/triage',
                        help='Path to pawpaw program bundle')
    parser.add_argument('--batch', action='store_true',
                        help='Use batch_call API for all tests')
    args = parser.parse_args()
    
    # Find all testbed scripts
    testbed_dir = Path(__file__).parent
    testbeds = sorted(testbed_dir.glob('testbed_*.py'))
    
    if not testbeds:
        print("No testbed scripts found!")
        return 1
    
    print(f"Found {len(testbeds)} testbed(s)")
    print(f"Bundle: {args.bundle}")
    print(f"Batch mode: {args.batch}")
    print()
    
    results: List[TestbedResult] = []
    
    # Run each testbed
    for testbed in testbeds:
        print(f"\n{'='*80}")
        print(f"Running: {testbed.name}")
        print('='*80)
        
        result = run_testbed(str(testbed), args.bundle, args.batch)
        results.append(result)
        
        # Show output
        print(result.output)
    
    # Print summary
    print_summary(results)
    
    # Return overall success
    all_passed = all(r.exit_code == 0 for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
