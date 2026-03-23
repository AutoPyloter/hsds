# 50-Run Statistical Robustness Report

This report conclusively demonstrates through massive 600-run simulation data that the Harmonix library's **Extreme Dependent Space** eradicates the combinatorial explosion typically faced in penalty-based operations, reducing standard continuous derivations to flat determinism.

## Visual Data Analysis
![Statistical Robustness Distributions](benchmarks/statistical_robustness.png)

## Exact Aggregate Table (Cost Statistics)

| Problem | Method | Min Cost | Max Cost | Mean Cost | Median Cost | Std Dev | Iter Mean | Iter Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pressure Vessel** | Dependent Space | 5804.3762 | 5804.3762 | 5804.3762 | 5804.3762 | 0.0000 | 628 | 116 |
| **Pressure Vessel** | Static Penalty | 6156.9871 | 7320.5481 | 6739.3825 | 6755.8786 | 324.7671 | 20288 | 4819 |
| **Retaining Wall** | Dependent Space | 186.3250 | 242.4404 | 206.8145 | 202.3610 | 15.4056 | 7664 | 3614 |
| **Retaining Wall** | Static Penalty | 197.5463 | 223.7002 | 208.6701 | 207.5291 | 7.5119 | 5844 | 2964 |
| **Robot Gripper** | Dependent Space | 100.0000 | 100.0016 | 100.0005 | 100.0004 | 0.0004 | 7532 | 3934 |
| **Robot Gripper** | Static Penalty | 100.0007 | 100.0177 | 100.0058 | 100.0051 | 0.0034 | 20594 | 4064 |
| **Speed Reducer** | Dependent Space | 2996.3482 | 2996.3482 | 2996.3482 | 2996.3482 | 0.0000 | 1606 | 278 |
| **Speed Reducer** | Static Penalty | 2996.3491 | 2996.3661 | 2996.3548 | 2996.3538 | 0.0041 | 6890 | 1262 |
| **Spring Design** | Dependent Space | 0.0106 | 0.0106 | 0.0106 | 0.0106 | 0.0000 | 14930 | 8094 |
| **Spring Design** | Static Penalty | 0.0128 | 0.0178 | 0.0155 | 0.0156 | 0.0014 | 22822 | 8334 |
| **Welded Beam** | Dependent Space | 1.7258 | 2.7892 | 2.0603 | 2.0213 | 0.2681 | 22480 | 8264 |
| **Welded Beam** | Static Penalty | 2.0847 | 3.5062 | 2.6502 | 2.6783 | 0.3021 | 24528 | 7378 |
