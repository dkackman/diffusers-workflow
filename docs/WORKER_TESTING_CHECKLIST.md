# Worker Implementation - Testing Checklist

## Pre-Testing Setup

- [x] All Python files compile without syntax errors
- [x] Basic worker tests pass (4/4)
- [x] Documentation created

## Manual Testing Checklist

### Basic Functionality

- [ ] Start REPL: `python -m dw.repl`
- [ ] Load a workflow: `load FluxDev` (or any available workflow)
- [ ] Set arguments: `arg prompt="test"`
- [ ] First run: `run` (should start worker, load models)
- [ ] Second run: `run` (should reuse cached models - MUCH faster)
- [ ] Third run: `run` (verify consistent behavior)

### Memory Monitoring

- [ ] Check memory: `memory` (should show GPU stats)
- [ ] Run workflow multiple times
- [ ] Check memory after each run (watch for growth)
- [ ] Clear memory: `clear`
- [ ] Check memory again: `memory` (should show cleared state)
- [ ] Run workflow again (should reload models)

### File Change Detection

- [ ] Load workflow: `load FluxDev`
- [ ] Run once: `run`
- [ ] Edit workflow file (change any value)
- [ ] Run again: `run` (should detect change and reload)
- [ ] Check output says "Workflow file changed - reloading models..."

### Workflow Switching

- [ ] Load workflow A: `load FluxDev`
- [ ] Run: `run`
- [ ] Load workflow B: `load sdxl`
- [ ] Check worker shutdown message
- [ ] Run: `run` (should start new worker)
- [ ] Load workflow A again: `load FluxDev`
- [ ] Run: `run` (should shutdown and restart worker)

### Error Handling

- [ ] Load valid workflow: `load FluxDev`
- [ ] Set invalid argument (if possible)
- [ ] Run: `run` (should show error, worker stays alive)
- [ ] Fix argument
- [ ] Run again: `run` (should work, using cached models)

### Graceful Shutdown

- [ ] Load and run a workflow
- [ ] Exit REPL: `exit`
- [ ] Check for "Shutting down worker process..." message
- [ ] Check for "Goodbye!" message
- [ ] Verify no zombie processes: `ps aux | grep worker`

### Edge Cases

- [ ] Start REPL
- [ ] Run without loading workflow: `run` (should error gracefully)
- [ ] Load workflow but don't run
- [ ] Exit: `exit` (should not crash)
- [ ] Start REPL
- [ ] Check memory without worker: `memory` (should say no worker)
- [ ] Clear without worker: `clear` (should say no worker)

### Performance Validation

- [ ] Load a large model workflow (e.g., FluxDev, SDXL)
- [ ] Time first run: `time` in shell or note start/end time
- [ ] Note model loading time (30-60 seconds typical)
- [ ] Time second run with same workflow
- [ ] Verify second run is significantly faster (2-4x)
- [ ] Time should be close to inference-only time

### Memory Leak Testing

- [ ] Load workflow
- [ ] Run 10 times in a row
- [ ] Check memory after each run: `memory`
- [ ] Memory should be relatively stable (small growth <1GB is OK)
- [ ] If growth >1GB per run, there's a leak
- [ ] Use `clear` to free memory
- [ ] Verify memory drops to near-zero

### Long-Running Workflow

If you have a slow workflow (>2 minutes):
- [ ] Load slow workflow
- [ ] Run: `run`
- [ ] Verify streaming output appears
- [ ] Verify timeout doesn't trigger (<5 minutes)
- [ ] Verify completion message

### Stress Testing

- [ ] Run same workflow 50+ times
- [ ] Check for any crashes
- [ ] Check for memory stability
- [ ] Check for any zombie processes
- [ ] Worker should stay alive entire time

## Expected Behaviors

### First Run
```
dw> run
Starting worker process...
Worker process started
Running workflow: FluxDev
Workflow file changed - reloading models...
Loading workflow from examples/FluxDev.json
Models loaded for workflow: FluxDev
Executing workflow: FluxDev
[... workflow output ...]
GPU Memory Status:
  Device: [GPU name]
  Allocated: [X] MB
  Reserved: [Y] MB
  ...
Workflow completed successfully
(Workflow has been executed 1 time(s) in this session)
```

### Subsequent Run (Cached)
```
dw> run
Running workflow: FluxDev
Reusing loaded models from cache
Executing workflow: FluxDev
[... workflow output - FASTER! ...]
GPU Memory Status:
  [similar memory usage]
Workflow completed successfully
(Workflow has been executed 2 time(s) in this session)
```

### File Change Detected
```
dw> run
Running workflow: FluxDev
Workflow file changed - reloading models...
Loading workflow from examples/FluxDev.json
[... reload process ...]
```

### Clear Memory
```
dw> clear
Clearing GPU memory...
GPU Memory Status:
  Device: [GPU name]
  Allocated: 0.0 MB
  Reserved: 0.0 MB
  ...
  Runs in this session: 0
GPU memory cleared successfully
```

### Memory Status
```
dw> memory
GPU Memory Status:
  Device: NVIDIA GeForce RTX 4090
  Allocated: 8234.5 MB
  Reserved: 8456.2 MB
  Free: 15234.8 MB
  Total: 24000.0 MB
  Runs in this session: 5
```

## Known Limitations

- Worker has 5-minute timeout per execution
- Single workflow cached at a time
- Worker restarts on workflow change (by design)
- Memory monitoring only available when worker running

## Troubleshooting

### Worker won't start
- Check CUDA installation: `nvidia-smi`
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Check for errors in terminal output

### Memory keeps growing
- Expected: Small growth (<500MB) is normal
- Concerning: Growth >1GB per run indicates leak
- Solution: Use `clear` command
- If persists: File bug report with workflow

### Worker crashes
- Check terminal for error messages
- Check workflow JSON is valid
- Try simpler workflow to isolate issue
- Restart REPL and try again

### Slow performance
- First run always slow (loading models)
- Subsequent runs should be 2-4x faster
- If all runs slow, worker may not be caching
- Check for file change messages

## Success Criteria

✅ All basic functionality tests pass  
✅ Memory monitoring works  
✅ File change detection works  
✅ Workflow switching works  
✅ Error handling is graceful  
✅ Second run is 2-4x faster than first  
✅ Memory stays stable over 10+ runs  
✅ No zombie processes after exit  

## Post-Testing

After completing checklist:
- [ ] Document any issues found
- [ ] Note any performance metrics
- [ ] Save example output
- [ ] Update documentation if needed

## Quick Start for Testing

```bash
# Terminal 1 - Start REPL
cd /home/don/diffusers-workflow
source venv/bin/activate
python -m dw.repl

# In REPL
dw> load FluxDev
dw> arg prompt="a beautiful sunset"
dw> run
# Wait for completion, note time
dw> run
# Should be MUCH faster!
dw> memory
dw> clear
dw> exit

# Terminal 2 - Monitor GPU (optional)
watch -n 1 nvidia-smi
```

## Automated Test

Quick automated test:
```bash
python test_worker.py
```

Should see:
```
Test 1: Worker lifecycle ✓
Test 2: Memory status reporting ✓
Test 3: Memory clearing ✓
Test 4: Simple workflow execution ✓ (if test.json exists)
Results: 4 passed, 0 failed
```
