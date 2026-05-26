# Hints

- Before Iter 1, run `ncu` on the baseline kernel to guide the first direction.
  If `ncu` is unavailable (driver mismatch, missing toolkit), note that in
  `ITERATIONS.md` and proceed with analytical reasoning + runtime stats (min vs
  mean) — don't skip iter-1 setup just because the profiler isn't runnable.
- If 3 consecutive iterations show no improvement, run `ncu` to re-profile, use WebSearch for new ideas, and review `ITERATIONS.md` for patterns. Plan before continuing.

<!-- Add directives below, for example:
- Optimization constraints or focus areas
- Strategies to try or avoid
- Agent behavior controls
- Dependency policies (e.g., "Do not install any packages") -->
