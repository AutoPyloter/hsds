# Welded Beam Design — Post-Run Analysis

## Performance Evaluation

The Welded Beam problem demonstrates the fundamental advantage of embedding physical constraints into the search space.

### Methodology Comparison

| Metric | Static Penalty | Dependent Space | Improvement |
|---|---|---|---|
| **Best Cost** | 2.5141 | 2.1669 | **~14% better** |
| **Execution Time** | 1.40s | 1.39s | Comparable |
| **Total Penalty** | 0.0 | 0.0 | Both converged |

### Engineering Insights
1. **Convergence Speed:** The Static Penalty approach remains trapped in the infeasible region with high penalty scores for the first 10,000 iterations. Conversely, the Extreme Dependent Space approach produces almost exclusively valid designs right from the first iteration.
2. **Space Reduction:** The buckling load and shear stress constraints are highly restrictive. By enforcing `tau <= tau_max` and `sigma <= sigma_max` logically through lambda functions, the algorithm prunes out geometrically unfavorable beams before evaluating their cost.
3. **Execution Edge:** Although the Extreme Dependent Space utilizes dynamic lambda rules to restrict the search space, it maintains a highly competitive execution time (reducing computational overhead by ~0.5%) by bypassing unnecessary penalty evaluations. However, the most significant advantage is the ~14% reduction in the final design cost.

### Visualization Guide
When observing the `welded_beam_comparison.png` graph:
- **Absence of initial spikes = Elimination of physically impossible designs.** The green dashed line (Dependent Space) immediately begins exploring the feasible cost block (under 6.0), entirely bypassing the massive structural failure region that the Static Penalty (red solid line) struggles through initially.
- **Steep descent = Rapid convergence.** The Dependent Space configuration drops precipitously toward 2.2 within the first 5,000 iterations, confirming exceptionally high exploitation efficiency.
