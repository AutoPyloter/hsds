# Retaining Wall Optimization — Engineering Analysis

## Geotechnical and Structural Synthesis

The Retaining Wall benchmark evaluates a highly complex interacting network of physics. Geotechnical active earth pressures ($P_a$) dictate mathematical failures in sliding and overturning stability, while ACI 318 code requirements command exact dimensions to resist shear forces ($V_u$) and flexural moments ($M_u$), alongside spacing ($\rho$) regulations on over 400 distinct rebar catalogs.

### Methodology Comparison

| Method                  | Best Cost | Feasibility Strategy | Time to Solution |
|-------------------------|-----------|----------------------|------------------|
| **Static Penalty**      | 219.461   | Blind Selection      | $\sim 1.22s$     |
| **Dependent Space**     | 190.080   | Exact Math Pruning   | $\sim 14.55s$    |

### Performance Insights
1. **Combinatorial Explosion (The Burden of Determinism):** The `static_penalty` baseline mathematically isolates diameter arrays (`12-32mm`) and count arrays (`2-41`) directly. Evaluating every integer combination blindly leads to a massive combinatorial explosion where the optimizer wastes $>90\%$ of its cycles testing physically impossible constraints (e.g. forcing 40 bars into a 1 meter space or entirely violating $\rho_{min}$). It stalled at a heavily suboptimal $219.46$ cost.
2. **True Zero-Penalty Formulation for Geotech:** The Dependent Space algorithm executes an exhaustive bisection loop to scan valid $x_{1,min}$ geometries that explicitly satisfy $FS_{overturning} \ge 2.5$ and bounds structural thicknesses statically against ACI shear capacities before a single rebar is evaluated.
3. **Native `ACIRebar` Spatial Integration:**
   Replacing the manual filters, the model maps two distinct dependent parameters: `dc_stem` and `dc_base` utilizing strictly the Harmonix `ACIRebar` spatial module. The generative bounds autonomously resolve ductility limits ($\rho_{min}$, $\rho_{max}$) and exact integer bar spacing constraints ($s \ge 25$mm) relative to the dynamic geotechnical depths ($d_{stem}$ and $d_{base}$). The minimal overhead explicitly ensures that 100% of tested reinforcing patterns are physically constructible without standard array penalties.

### Visual Implications
Observing `retaining_wall_comparison.png`, the dependency framework completely eliminates the volatile stochastic penalty plateaus visible in standard Harmony Search. It treats optimization not as a game of chance, but as a procedural algebraic traversal of valid geometries.
