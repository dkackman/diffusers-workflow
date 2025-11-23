#!/usr/bin/env python3
"""
Test script for worker-based REPL implementation.
This tests the persistent worker subprocess with GPU memory management.
"""

import sys
import os
import time
import multiprocessing

# CRITICAL: Set spawn method before any other imports that might use multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.worker import worker_main


def test_worker_lifecycle():
    """Test basic worker startup and shutdown"""
    print("Test 1: Worker lifecycle")
    print("-" * 50)
    
    cmd_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    
    worker = multiprocessing.Process(target=worker_main, args=(cmd_queue, res_queue, "INFO"))
    worker.start()
    
    print("✓ Worker started")
    
    # Test ping
    cmd_queue.put({'type': 'ping'})
    result = res_queue.get(timeout=5)
    assert result['type'] == 'pong'
    print(f"✓ Worker responded to ping (run_count: {result['run_count']})")
    
    # Test shutdown
    cmd_queue.put({'type': 'shutdown'})
    result = res_queue.get(timeout=5)
    assert result['type'] == 'shutdown_complete'
    worker.join(timeout=5)
    print("✓ Worker shutdown gracefully")
    
    print("✓ Test 1 passed\n")


def test_worker_memory_status():
    """Test memory status reporting"""
    print("Test 2: Memory status reporting")
    print("-" * 50)
    
    cmd_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    
    worker = multiprocessing.Process(target=worker_main, args=(cmd_queue, res_queue, "INFO"))
    worker.start()
    
    print("✓ Worker started")
    
    # Request memory status
    cmd_queue.put({'type': 'memory_status'})
    result = res_queue.get(timeout=5)
    
    assert result['type'] == 'memory_status'
    info = result['info']
    
    print(f"  GPU Available: {info['gpu_available']}")
    if info['gpu_available']:
        print(f"  Device: {info['gpu_device_name']}")
        print(f"  Allocated: {info['gpu_memory_allocated_mb']:.1f} MB")
        print(f"  Reserved: {info['gpu_memory_reserved_mb']:.1f} MB")
    
    print("✓ Memory status retrieved")
    
    # Shutdown
    cmd_queue.put({'type': 'shutdown'})
    res_queue.get(timeout=5)
    worker.join(timeout=5)
    print("✓ Worker shutdown")
    
    print("✓ Test 2 passed\n")


def test_worker_with_simple_workflow():
    """Test workflow execution if a simple workflow exists"""
    print("Test 3: Simple workflow execution")
    print("-" * 50)
    
    # Check if test workflow exists
    test_workflow = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dw', 'workflows', 'test.json'
    )
    
    if not os.path.exists(test_workflow):
        print("⊘ Test workflow not found, skipping")
        print()
        return
    
    cmd_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    
    worker = multiprocessing.Process(target=worker_main, args=(cmd_queue, res_queue, "INFO"))
    worker.start()
    
    print("✓ Worker started")
    
    # Execute workflow
    output_dir = './test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd_queue.put({
        'type': 'execute',
        'workflow_path': test_workflow,
        'arguments': {},
        'output_dir': output_dir,
        'log_level': 'INFO'
    })
    
    print("  Workflow execution started...")
    
    # Collect results
    success = False
    while True:
        result = res_queue.get(timeout=60)
        result_type = result.get('type')
        
        if result_type == 'output':
            print(f"  {result['message']}")
        elif result_type == 'workflow_loaded':
            print(f"  ✓ Workflow loaded: {result['workflow_name']}")
        elif result_type == 'success':
            print(f"  ✓ {result['message']}")
            success = True
            break
        elif result_type == 'error':
            print(f"  ✗ Error: {result['message']}")
            break
    
    if success:
        print("✓ Workflow executed successfully")
        
        # Execute again to test caching
        print("  Running workflow again to test caching...")
        cmd_queue.put({
            'type': 'execute',
            'workflow_path': test_workflow,
            'arguments': {},
            'output_dir': output_dir,
            'log_level': 'INFO'
        })
        
        while True:
            result = res_queue.get(timeout=60)
            result_type = result.get('type')
            
            if result_type == 'output':
                if 'cache' in result['message'].lower():
                    print(f"  ✓ {result['message']}")
            elif result_type == 'success':
                print(f"  ✓ Second run completed (run_count: {result['run_count']})")
                break
            elif result_type == 'error':
                print(f"  ✗ Error on second run: {result['message']}")
                break
    
    # Shutdown
    cmd_queue.put({'type': 'shutdown'})
    res_queue.get(timeout=5)
    worker.join(timeout=5)
    print("✓ Worker shutdown")
    
    print("✓ Test 3 passed\n")


def test_worker_clear_memory():
    """Test memory clearing"""
    print("Test 4: Memory clearing")
    print("-" * 50)
    
    cmd_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    
    worker = multiprocessing.Process(target=worker_main, args=(cmd_queue, res_queue, "INFO"))
    worker.start()
    
    print("✓ Worker started")
    
    # Clear memory
    cmd_queue.put({'type': 'clear_memory'})
    result = res_queue.get(timeout=10)
    
    assert result['type'] == 'memory_cleared'
    print("✓ Memory cleared successfully")
    
    info = result['info']
    if info['gpu_available']:
        print(f"  GPU memory after clear: {info['gpu_memory_allocated_mb']:.1f} MB")
    
    # Shutdown
    cmd_queue.put({'type': 'shutdown'})
    res_queue.get(timeout=5)
    worker.join(timeout=5)
    print("✓ Worker shutdown")
    
    print("✓ Test 4 passed\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("Worker Process Tests")
    print("=" * 50 + "\n")
    
    tests = [
        test_worker_lifecycle,
        test_worker_memory_status,
        test_worker_clear_memory,
        test_worker_with_simple_workflow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
