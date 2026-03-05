# Lesson 12: Hardening, Testing, and Submission Readiness

## Learning Objectives

By the end of this lesson, you will:

- Validate full-project reliability with tests
- Check reproducibility and artifact consistency
- Prepare a clean submission package for demo/review
- Document assumptions, limits, and deployment notes

## Build Tasks

1. Run full test suite and fix regressions introduced by your implementation.
2. Validate artifact compatibility:
   - selected features
   - scaler input dimensions
   - model input dimensions
3. Run at least one end-to-end training + evaluation + demo path.
4. Prepare concise documentation and demo command set.

## Exercise

### Exercise 1 (Medium)

Create a one-command smoke test script for your rebuilt project.

### Exercise 2 (Hard)

Create a reproducibility check: rerun with same seed and compare key metrics.

### Exercise 3 (Hard)

Write a final project summary with architecture, metrics, and limitations.

## Verification

```bash
python -m pytest tests/ -v
python quick_cascaded_demo.py
python evaluate.py --test-data <your_test_csv>
```

## Solution Reference

- `tests/`
- `PROJECT_SETUP_AND_USAGE_GUIDE.md`
- `COMPLETE_PROJECT_EXPLANATION.md`
- `CLEANUP_AND_SUBMISSION_SUMMARY.md`
- `RUN_ALL_COMMANDS.md`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Final Readiness Checklist

- [ ] All module-level checks pass
- [ ] Full test suite runs
- [ ] Demo command works reliably
- [ ] Reports generated and readable
- [ ] You can explain architecture end-to-end without notes
