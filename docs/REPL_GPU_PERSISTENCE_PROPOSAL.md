# REPL GPU Memory Persistence - Solution Proposals

## Problem Statement

The REPL currently spawns a new subprocess for each workflow execution, which causes:
- **GPU models to be unloaded** after each run
- **High startup cost** - models must be reloaded from disk every time
- **Slow iteration** - defeats the REPL's purpose of fast experimentation

**Goal:** Keep models loaded in GPU memory across multiple runs of the same workflow file.

## Current Architecture Analysis

```
REPL Process (repl.py)
    └─> Subprocess (run.py) for each execution
            └─> Load models → Execute → Exit → GPU cleared
```

**Key observations:**
- Subprocess provides memory isolation (prevents leaks in main process)
- Security validation happens before subprocess spawn
- Real-time output streaming works well
- Each execution is stateless

## Proposed Solutions

### Option 1: Persistent Worker Subprocess (RECOMMENDED)

Keep the subprocess alive between runs, restart only when workflow file changes.

#### Architecture
```python
REPL Process (repl.py)
    └─> Worker Process (new: worker.py)
            - Starts on first 'run' command
            - Loads workflow and models into GPU
            - Accepts commands via queue/pipe
            - Monitors workflow file for changes
            - Exits only when:
                * Workflow file changes (detected via hash/mtime)
                * User loads different workflow
                * User issues 'clear' command
                * Worker crashes
```

#### Implementation Details

**New module: `dw/worker.py`**
```python
class WorkflowWorker:
    """Persistent worker process that keeps models loaded"""
    
    def __init__(self, command_queue, result_queue):
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.current_workflow = None
        self.workflow_hash = None
        self.loaded_pipelines = {}
        self.shared_components = {}
    
    def run(self):
        """Main worker loop - processes commands until shutdown"""
        while True:
            try:
                command = self.command_queue.get()
                
                if command['type'] == 'execute':
                    self._execute_workflow(command)
                elif command['type'] == 'shutdown':
                    self._cleanup()
                    break
                elif command['type'] == 'ping':
                    self.result_queue.put({'status': 'alive'})
                    
            except Exception as e:
                self.result_queue.put({'status': 'error', 'error': str(e)})
    
    def _execute_workflow(self, command):
        """Execute workflow, reusing loaded pipelines if possible"""
        workflow_path = command['workflow_path']
        arguments = command['arguments']
        output_dir = command['output_dir']
        
        # Check if workflow changed
        current_hash = self._compute_file_hash(workflow_path)
        
        if current_hash != self.workflow_hash:
            # Workflow changed - cleanup and reload
            self._cleanup()
            self.current_workflow = workflow_from_file(workflow_path, output_dir)
            self.workflow_hash = current_hash
            self.result_queue.put({'status': 'workflow_loaded'})
        
        # Execute workflow (models stay in GPU between runs)
        try:
            self.current_workflow.run(arguments, self.loaded_pipelines)
            self.result_queue.put({'status': 'success'})
        except Exception as e:
            self.result_queue.put({'status': 'error', 'error': str(e)})
    
    def _cleanup(self):
        """Clear GPU memory and reset state"""
        import gc
        import torch
        
        self.loaded_pipelines.clear()
        self.shared_components.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _compute_file_hash(self, path):
        """Compute hash of workflow file to detect changes"""
        import hashlib
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
```

**Modified REPL: `dw/repl.py`**
```python
class DiffusersWorkflowREPL(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.worker_process = None
        self.command_queue = None
        self.result_queue = None
        self.current_workflow_path = None
        # ... existing code ...
    
    def _ensure_worker(self):
        """Start worker process if not running"""
        if self.worker_process is None or not self.worker_process.is_alive():
            from multiprocessing import Process, Queue
            self.command_queue = Queue()
            self.result_queue = Queue()
            
            self.worker_process = Process(
                target=worker_main,
                args=(self.command_queue, self.result_queue)
            )
            self.worker_process.start()
    
    def do_run(self, arg):
        """Run workflow using persistent worker"""
        if not self.current_workflow:
            print("Error: No workflow loaded")
            return
        
        try:
            self._ensure_worker()
            
            # Send execute command
            self.command_queue.put({
                'type': 'execute',
                'workflow_path': self.current_workflow.file_spec,
                'arguments': self.workflow_args,
                'output_dir': self.globals['output_dir']
            })
            
            # Stream results
            while True:
                result = self.result_queue.get()
                
                if result['status'] == 'workflow_loaded':
                    print("Workflow loaded into worker")
                elif result['status'] == 'success':
                    print("Workflow completed successfully")
                    break
                elif result['status'] == 'error':
                    print(f"Error: {result['error']}")
                    break
                elif result['status'] == 'output':
                    print(result['line'])
                    
        except Exception as e:
            print(f"Error: {e}")
    
    def do_load(self, arg):
        """Load new workflow - shuts down worker if workflow changed"""
        old_path = self.current_workflow.file_spec if self.current_workflow else None
        
        # ... existing load logic ...
        
        # If workflow changed, shutdown worker (will restart on next run)
        if old_path != self.current_workflow.file_spec:
            self._shutdown_worker()
    
    def _shutdown_worker(self):
        """Gracefully shutdown worker process"""
        if self.worker_process and self.worker_process.is_alive():
            self.command_queue.put({'type': 'shutdown'})
            self.worker_process.join(timeout=5)
            if self.worker_process.is_alive():
                self.worker_process.terminate()
    
    def do_clear(self, arg):
        """Clear GPU memory by restarting worker"""
        self._shutdown_worker()
        print("GPU memory cleared")
    
    def do_exit(self, arg):
        """Exit REPL"""
        self._shutdown_worker()
        return True
```

#### Pros
✅ **GPU persistence** - Models stay loaded between runs  
✅ **Memory isolation** - Worker crash doesn't affect REPL  
✅ **Automatic cleanup** - Worker restarts when workflow changes  
✅ **Security maintained** - All validation happens before worker spawn  
✅ **Clean separation** - REPL UI separate from execution  
✅ **Explicit control** - User can `clear` to free GPU manually  

#### Cons
❌ **Complexity** - Inter-process communication adds complexity  
❌ **Debugging** - Harder to debug worker process  
❌ **State management** - Must track worker lifecycle  

#### Implementation Effort
- **New code:** ~300 lines (worker.py + queue handling)
- **Modified code:** ~200 lines in repl.py
- **Testing:** Need multiprocessing test fixtures
- **Time estimate:** 4-6 hours

---

### Option 2: In-Process Execution with Aggressive Cleanup

Move workflow execution into the REPL process with careful memory management.

#### Architecture
```python
REPL Process (repl.py)
    └─> Direct execution of workflow.run()
            - Models loaded once per workflow file
            - Cached until workflow file changes
            - Aggressive cleanup between runs
            - Manual gc.collect() after each step
```

#### Implementation Details

**Modified REPL: `dw/repl.py`**
```python
class DiffusersWorkflowREPL(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.loaded_pipelines = {}  # Cache pipelines per workflow
        self.workflow_file_hash = None
        self.shared_components = {}
        # ... existing code ...
    
    def _compute_file_hash(self, path):
        """Track workflow file changes"""
        import hashlib
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _clear_gpu_memory(self):
        """Aggressive GPU memory cleanup"""
        import gc
        import torch
        
        self.loaded_pipelines.clear()
        self.shared_components.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    def do_load(self, arg):
        """Load workflow and clear cache if file changed"""
        # ... existing validation ...
        
        new_hash = self._compute_file_hash(file_path)
        
        if new_hash != self.workflow_file_hash:
            # Workflow changed - clear everything
            self._clear_gpu_memory()
            self.workflow_file_hash = new_hash
        
        self.current_workflow = workflow_from_file(file_path, output_dir)
        print(f"Loaded workflow: {self.current_workflow.name}")
    
    def do_run(self, arg):
        """Run workflow directly in-process"""
        if not self.current_workflow:
            print("Error: No workflow loaded")
            return
        
        try:
            print(f"Running workflow: {self.current_workflow.name}")
            
            # Execute directly in this process
            self.current_workflow.run(
                self.workflow_args,
                self.loaded_pipelines  # Reuse cached pipelines
            )
            
            # Cleanup after run (but keep models loaded)
            import gc
            gc.collect()
            
            print("Workflow completed successfully")
            
        except Exception as e:
            print(f"Error: {e}")
            # On error, clear everything
            self._clear_gpu_memory()
    
    def do_clear(self, arg):
        """Manually clear GPU memory"""
        self._clear_gpu_memory()
        print("GPU memory cleared")
```

**Modified Workflow: `dw/workflow.py`**
```python
class Workflow:
    def run(self, arguments, previous_pipelines=None):
        """Modified to accept and reuse pipelines"""
        
        # Use provided pipelines or create new dict
        if previous_pipelines is None:
            previous_pipelines = {}
        
        # ... existing variable processing ...
        
        # Track if we're reusing pipelines
        reusing_pipelines = len(previous_pipelines) > 0
        
        for step_def in steps:
            step_name = step_def["name"]
            
            # Check if pipeline already exists
            if step_name in previous_pipelines and "pipeline" in step_def:
                logger.debug(f"Reusing cached pipeline for step: {step_name}")
                action = previous_pipelines[step_name]
            else:
                action = self.create_step_action(
                    step_def,
                    shared_components,
                    previous_pipelines,
                    default_seed,
                    device_identifier,
                )
            
            # ... execute step ...
            
            # Cleanup between steps
            import gc
            gc.collect()
        
        # Don't clear pipelines - they'll be reused
        return previous_results
```

#### Memory Leak Prevention Strategy

1. **Explicit tensor cleanup:**
```python
# After each step, ensure tensors are released
for key, value in step_results.items():
    if isinstance(value, torch.Tensor):
        value.detach()
        del value
```

2. **Context managers for temporary objects:**
```python
with torch.inference_mode():
    output = pipeline(**arguments)
# Automatically releases computation graph
```

3. **Periodic forced cleanup:**
```python
# Every N runs, do aggressive cleanup
self.run_count += 1
if self.run_count % 10 == 0:
    self._clear_gpu_memory()
```

4. **Monitor memory growth:**
```python
def _check_memory_health(self):
    """Warn if memory keeps growing"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        if allocated > self.last_memory * 1.5:
            print("Warning: Memory usage increasing - consider 'clear' command")
        self.last_memory = allocated
```

#### Pros
✅ **Simplicity** - No multiprocessing complexity  
✅ **GPU persistence** - Models stay loaded between runs  
✅ **Direct debugging** - Easier to debug in same process  
✅ **Lower overhead** - No IPC serialization  
✅ **Shared memory** - No data copying between processes  

#### Cons
❌ **Memory leak risk** - Python/PyTorch memory management is tricky  
❌ **Process stability** - GPU error could crash REPL  
❌ **Shared state** - Harder to ensure clean state between runs  
❌ **Memory monitoring** - User must watch for leaks manually  

#### Implementation Effort
- **New code:** ~100 lines (memory management utilities)
- **Modified code:** ~150 lines in repl.py, ~50 lines in workflow.py
- **Testing:** Need memory leak tests
- **Time estimate:** 2-3 hours

---

### Option 3: Hybrid Approach with Model Cache Service

Separate model loading from execution - use a persistent model cache service.

#### Architecture
```python
REPL Process (repl.py)
    └─> Model Cache Service (cache_server.py)
    |       - Long-running process
    |       - Loads and caches models by name
    |       - Provides model references via shared memory
    |
    └─> Execution Subprocess (run.py)
            - Short-lived per execution
            - Requests models from cache
            - Executes workflow
            - Exits without unloading models
```

#### Implementation Details

**New module: `dw/model_cache_server.py`**
```python
class ModelCacheServer:
    """Long-running service that manages model cache"""
    
    def __init__(self):
        self.cache = {}  # model_name -> loaded_model
        self.ref_counts = {}  # Track usage
        
    def get_model(self, model_name, load_fn):
        """Get model from cache or load it"""
        if model_name not in self.cache:
            self.cache[model_name] = load_fn()
        
        self.ref_counts[model_name] = self.ref_counts.get(model_name, 0) + 1
        return self.cache[model_name]
    
    def release_model(self, model_name):
        """Decrement reference count"""
        self.ref_counts[model_name] -= 1
        
        # Could implement LRU eviction here
        if self.ref_counts[model_name] <= 0:
            # Keep it cached anyway
            pass
```

**Note:** This requires shared GPU memory between processes, which is complex with PyTorch. Would need to use `torch.multiprocessing` with careful `share_memory_()` calls.

#### Pros
✅ **Clean separation** - Models isolated from execution  
✅ **Smart caching** - Can implement LRU, size limits  
✅ **Multiple workflows** - Cache shared across different workflows  
✅ **Robust** - Execution crash doesn't lose models  

#### Cons
❌ **High complexity** - Shared memory with PyTorch is difficult  
❌ **Platform-specific** - CUDA shared memory has limitations  
❌ **Serialization overhead** - Models must be shareable  
❌ **Debugging nightmare** - Three process types to coordinate  

#### Implementation Effort
- **New code:** ~500 lines (cache server + client)
- **Modified code:** ~200 lines
- **Testing:** Complex multiprocess testing
- **Time estimate:** 12-16 hours
- **Risk:** High - PyTorch shared memory is unreliable

---

## Comparison Matrix

| Feature | Option 1: Persistent Worker | Option 2: In-Process | Option 3: Cache Service |
|---------|----------------------------|---------------------|------------------------|
| **GPU Persistence** | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **Memory Isolation** | ✅ Yes | ❌ No | ✅ Yes |
| **Complexity** | 🟨 Medium | ✅ Low | ❌ High |
| **Debugging** | 🟨 Medium | ✅ Easy | ❌ Hard |
| **Leak Risk** | ✅ Low | 🟨 Medium | ✅ Low |
| **Crash Safety** | ✅ Yes | ❌ No | ✅ Yes |
| **Implementation Time** | 🟨 4-6 hours | ✅ 2-3 hours | ❌ 12-16 hours |
| **Maintenance** | 🟨 Medium | ✅ Low | ❌ High |
| **Cross-workflow Cache** | ❌ No | ❌ No | ✅ Yes |

---

## Recommendation

**Option 1: Persistent Worker Subprocess** is the best choice because:

1. **Balanced tradeoffs** - Good persistence without high complexity
2. **Crash safety** - Worker crashes don't kill REPL
3. **Memory isolation** - Leaks contained to worker
4. **Clear lifecycle** - Easy to understand when worker restarts
5. **Security maintained** - Existing validation still applies
6. **Proven pattern** - Similar to Jupyter kernel architecture

**Quick Win Alternative:** Implement Option 2 first (2-3 hours) as a proof-of-concept to validate GPU persistence works as expected, then migrate to Option 1 for production use.

---

## Implementation Roadmap

### Phase 1: Proof of Concept (Option 2)
1. Move execution into REPL process
2. Add file hash tracking
3. Implement basic memory cleanup
4. Test with simple workflows
5. Monitor for memory leaks

### Phase 2: Production Implementation (Option 1)
1. Create `worker.py` module
2. Implement command queue protocol
3. Add output streaming from worker
4. Implement worker lifecycle management
5. Add `clear` command for manual cleanup
6. Test with complex workflows

### Phase 3: Enhancements
1. Add memory usage monitoring
2. Implement auto-cleanup thresholds
3. Add worker health checks
4. Support graceful worker restart
5. Add metrics (run count, memory usage, etc.)

---

## Additional Considerations

### Security
- All validation happens in REPL before sending to worker
- Worker only receives pre-validated paths and arguments
- Same security model as current subprocess approach

### Error Handling
- Worker sends detailed error messages back to REPL
- REPL can restart worker on fatal errors
- User sees clear error messages in both cases

### User Experience
- Transparent to user - just faster on subsequent runs
- `clear` command for manual control
- Status messages ("Using cached models", "Loading models...")
- Memory usage indicators

### Testing Strategy
- Unit tests for worker protocol
- Integration tests for REPL-worker communication
- Memory leak tests (run same workflow 100x)
- Stress tests (large models, many steps)
- Error recovery tests (worker crashes)
