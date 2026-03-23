# Speed Reducer Design — Post-Run Analysis

## Performance Evaluation

The Speed Reducer Design problem acts as a definitive test of extreme domain reduction due to the highly interconnected nature of its 11 constraints.

### Methodology Comparison

| Method                  | Best Cost | Geometric Bounds Enforced |
|-------------------------|-----------|---------------------------|
| **Static Penalty**      | 2996.35   | None (All in Penalty)     |
| **Semi-Dependent**      | 2996.35   | Face Width ($b$)          |
| **Full Parametric**     | 2996.34   | Complete Hierarchy (11)   |

### Engineering Insights
1. **Mathematical Precision:** Because the optimum of this problem lies explicitly on the constraint boundaries rather than within a wide feasible interior, all methods converge to effectively the identical literature optimum (~2996.35).
2. **The Bottleneck Abstraction:** The `semi_dependent` configuration successfully absorbed 6 constraints directly into the boundary evaluations of the Face Width ($b \rightarrow x_1$). By procedurally ensuring that generated widths were always compatible with the sampled module ($m$), teeth ($z$), and shafts ($l_1, l_2, d_1, d_2$), the optimizer substantially bypassed the most mathematically restrictive subspace.
3. **Zero-Penalty Mastery:** The `full_parametric_extreme` formulation achieved a 100% dependency resolution. By hierarchically structuring the parameter generation ($m \rightarrow l_1, l_2 \rightarrow z \rightarrow d_1 \rightarrow d_2 \rightarrow b$), the algorithm eliminated all 11 constraints from the evaluation penalty. This demonstrates the ultimate capability of the Harmonix library to transmute standard optimization blockades directly into searchable topology.

### Visualization Guide
When observing the `speed_reducer_comparison.png` graph:
- **Absence of initial spikes:** Represents the immediate elimination of physically impossible designs, bypassing the exploratory penalty evaluation loops entirely in the dependent architectures.
- **Steep descent profile:** The Semi-Dependent and Full Parametric configurations (dashed and dotted lines) exhibit an inherently faster stabilization curve. They strictly exploit valid geometric combinations, ignoring the massive infeasible regions that typically stall standard optimization samplers.
