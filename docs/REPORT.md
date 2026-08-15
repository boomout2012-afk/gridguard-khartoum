# GridGuard Khartoum final report

**Full title:** GridGuard Khartoum: PPO-Assisted Load Shedding and Energy Management for
Developing Countries

**Document type:** Graduation Project Final Report

**Authors:**

- Murtada Asim Ibrahim Bakhit (Index 18-317)
- Suliman Elawad Suliman Babiker (Index 18-310)
- Abdalhay Abass Mobarak Abass (Index 15-306)

**Date:** August 2026

## Abstract

GridGuard Khartoum is an offline, GPU-first research simulation for proposal-aligned load
shedding across seven Khartoum State localities. It compares a frozen PPO controller with Fixed
Rotation on identical synthetic 24-hour shortage scenarios. The release preserves the accepted
aggregate comparison and regenerates auditable paired episode evidence from the frozen PPO
checkpoint and reviewed simulator. Across 96 independent base draws replayed under three
shortage levels, PPO improves reward, critical-service protection, utilization, fairness, served
energy, curtailed energy, and switching rate; both controllers have zero recorded power-balance
violations under the locked tolerance. A native offline desktop application provides an
interactive schematic map and deterministic live replay. The work is decision-support research,
not a deployed electrical-grid controller, and makes no field-performance claim.

## Assembly

The generated report uses this front matter followed by the committed chapters:

1. `docs/METHODOLOGY.md`
2. `docs/IMPLEMENTATION.md`
3. `docs/RESULTS.md`
4. `docs/REFERENCES.md`

Build with `uv run --extra docs python tools/build_report.py`. The output is
`output/pdf/gridguard-khartoum-final-report.pdf`.
