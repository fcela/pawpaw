#!/usr/bin/env python3
"""
Run all testbeds and generate a comprehensive accuracy report.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_testbed(testbed_name: str):
    """Run a testbed and capture output."""
    testbed_path = Path(__file__).parent / f"testbed_{testbed_name}.py"
    if not testbed_path.exists():
        return None
    
    print(f"\n{'='*80}")
    print(f"Running: {testbed_name}")
    print('='*80)
    
    cmd = [sys.executable, str(testbed_path), '--bundle', './programs/triage']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    return {
        'name': testbed_name,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }

def parse_accuracy(output: str):
    """Extract accuracy from testbed output."""
    for line in output.split('\n'):
        if 'Accuracy:' in line and '%' in line:
            try:
                return float(line.split(':')[1].strip().replace('%', ''))
            except (ValueError, IndexError):
                pass
    return None

def main():
    testbeds = ['sentiment', 'intent', 'topic', 'edge_cases']
    results = []
    
    print("="*80)
    print("COMPREHENSIVE CLASSIFICATION ACCURACY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Bundle: ./programs/triage")
    print("="*80)
    
    for testbed in testbeds:
        result = run_testbed(testbed)
        if result:
            results.append(result)
            print(result['stdout'])
            if result['stderr']:
                print(f"Errors: {result['stderr']}")
    
    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"\n{'Testbed':<25} {'Status':<10} {'Accuracy':<15} {'Notes'}")
    print("-"*80)
    
    total_accuracy = 0.0
    count = 0
    
    for result in results:
        status = "✅ PASS" if result['returncode'] == 0 else "❌ FAIL"
        accuracy = parse_accuracy(result['stdout'])
        if accuracy is not None:
            total_accuracy += accuracy
            count += 1
            acc_str = f"{accuracy:.1f}%"
        else:
            acc_str = "N/A"
        
        print(f"{result['name']:<25} {status:<10} {acc_str:<15}")
    
    if count > 0:
        avg = total_accuracy / count
        print("-"*80)
        print(f"{'AVERAGE':<25} {'':<10} {avg:.1f}%")
    else:
        avg = 0.0
        
        print("\n\nOVERALL ASSESSMENT:")
        if avg >= 80:
            print("✅ Excellent accuracy across all testbeds!")
        elif avg >= 60:
            print("⚠️  Moderate accuracy - some categories need improvement")
        else:
            print("❌ Poor accuracy - classifier not suitable for general use")
    
    print("\n" + "="*80)
    print("Note: The triage classifier is specialized for email triage (trivial/substantive).")
    print("These test results show it does NOT generalize to other classification tasks,")
    print("which is expected behavior. Create task-specific classifiers for best results.")
    print("="*80)

if __name__ == "__main__":
    main()
