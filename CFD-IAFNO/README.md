# CFD + IAFNO Cleanroom Airflow Simulation

This project combines OpenFOAM CFD, IAFNO neural-operator prediction, and Lagrangian particle tracing for cleanroom airflow simulation.

## Project Goal

The goal is to build a fast surrogate model for cleanroom airflow prediction. OpenFOAM generates physically based CFD velocity fields, while IAFNO learns to predict future 3D velocity fields from previous CFD data.

## Main Workflow

1. Generate cleanroom CFD data with OpenFOAM.
2. Export the velocity field `U`.
3. Train an IAFNO model on 3D velocity fields.
4. Predict future airflow fields.
5. Trace Lagrangian particles through the predicted field.
6. Add particle-energy loss to keep particle motion stable.
7. Visualize airflow, particles, and CAD obstacles.

## Main Equation

Eulerian velocity field:

```math
\mathbf{u}(x,y,z,t) = (u,v,w)
```

Lagrangian particle motion:

```math
\frac{d\mathbf{x}_p}{dt} = \mathbf{u}(\mathbf{x}_p,t)
```

Total training loss:

```math
\mathcal{L}_{total}
=
\mathcal{L}_{field}
+
\lambda_E \mathcal{L}_{energy}
+
\lambda_s \mathcal{L}_{smooth}
```

## Current Setup

- Velocity field only
- Temperature and humidity removed
- 3D resolution target: `42 × 42 × 42`
- Particle count target: `50`
- CFD solver: OpenFOAM
- Neural model: IAFNO
- Particle method: Lagrangian tracing

## Important Problems

- High resolution can crash the kernel because memory cost grows quickly.
- Particles can disappear if boundary conditions are not handled correctly.
- Particles can pass through obstacles if no obstacle collision check is used.
- Particle wiggle can come from unstable velocity prediction or noisy interpolation.

## Final Direction

The final system should compare OpenFOAM CFD particle motion with IAFNO-predicted particle motion and use physical losses to improve stability.
