# Pose4 = SE(2)×ℝ

Implementation: `strataslam_core/include/strataslam/geometry/pose4.hpp`, tests in
`strataslam_core/test/pose4_test.cpp`. Notation follows Solà, Deray, Atchuthan, *A micro Lie theory
for state estimation in robotics* (arXiv:1812.01537): right-perturbation `X ⊕ ξ = X·exp(ξ)`,
`Ad_X`, left/right Jacobians `J_l`, `J_r`.

## Group

`X = (θ, t)`, `θ ∈ (−π, π]`, `t ∈ ℝ³`. It acts on points as `x' = R_z(θ)x + t`; as a matrix it is
the SE(3) transform with rotation `R_z(θ)`.

- `X₁X₂ = (θ₁+θ₂, R_z(θ₁)t₂ + t₁)` (angle re-wrapped)
- `X⁻¹ = (−θ, −R_z(−θ)t)`

## Algebra, exp and log

`ξ = (φ, ρ) ∈ ℝ⁴`, `ρ = (ρx, ρy, ρz)`. Write `ρ_xy = (ρx, ρy)`.

```
exp(ξ) = (φ, ( V(φ)ρ_xy , ρz ))
V(φ)   = (1/φ) [[ sin φ,  −(1−cos φ) ],
                [ 1−cos φ,   sin φ   ]]        V(0) = I
log(X) = (θ, ( V(θ)⁻¹ t_xy , t_z ))
V(θ)⁻¹ = [[ c, θ/2 ], [ −θ/2, c ]],   c = (θ/2)·cot(θ/2)
```

`V⁻¹` follows from `det V = 2(1−cos φ)/φ²`. Numerics:

- `1−cos φ` is evaluated as `2 sin²(φ/2)` (no cancellation).
- For `|φ| < 1e-5` the second-order expansions are used: `sin φ/φ ≈ 1 − φ²/6`,
  `(1−cos φ)/φ ≈ φ/2 − φ³/24`, `c ≈ 1 − φ²/12 − φ⁴/720`.
- `log(exp(ξ)) = ξ` holds for `|φ| < π`. `V` is not periodic in φ, so for `|φ| ≥ π` the pair
  `exp`/`log` returns the representative with `φ ∈ (−π, π]`, as usual for SE(2).

## Adjoint

`X exp(ξ) X⁻¹ = exp(Ad_X ξ)`. With `J = [[0, −1], [1, 0]]` and `R = R_z(θ)|_{2×2}`:

```
Ad_X = [[ 1,        0,  0 ],
        [ −J t_xy,  R,  0 ],      (−J t_xy = (t_y, −t_x))
        [ 0,        0,  1 ]]
```

## Jacobians

`J_l(ξ) = ∫₀¹ Ad_{exp(sξ)} ds` (`exp(ξ+δ) ≈ exp(J_l δ)·exp(ξ)`) and `J_r(ξ) = J_l(−ξ)`
(`exp(ξ+δ) ≈ exp(ξ)·exp(J_r δ)`). Evaluating the integral:

```
J_l = [[ 1,                    0,   0,  0 ],
       [ ρx·d + ρy·e,          a,  −b,  0 ],
       [ ρy·d − ρx·e,          b,   a,  0 ],
       [ 0,                    0,   0,  1 ]]

a = sin φ/φ,  b = (1−cos φ)/φ,  d = (φ − sin φ)/φ²,  e = (1−cos φ)/φ²
J_l⁻¹ = [[ 1, 0, 0 ], [ −V⁻¹ q, V⁻¹, 0 ], [ 0, 0, 1 ]]      q = first column below the 1
```

`d` and `e` use series (to φ⁵ / φ⁴ terms) below `|φ| < 1e-2`, where the closed form would lose
digits to cancellation. Relations verified in tests: `J_l = Ad_{exp ξ} J_r`, `J_r(ξ) = J_l(−ξ)`,
`J J⁻¹ = I`, and both against central finite differences.

## GTSAM integration

`Pose4` derives from `gtsam::LieGroup<Pose4, 4>` and specialises `gtsam::traits<Pose4>`.
`retract(ξ) = X·exp(ξ)`, `localCoordinates(Y) = log(X⁻¹Y)`; `Expmap(ξ, H)` returns `J_r(ξ)` and
`Logmap(X, H)` returns `J_r⁻¹(log X)`, as GTSAM's chart machinery expects.

## Conversion from SE(3)

`Pose4::FromPose3` returns `θ = atan2(R₁₀, R₀₀)` if both |roll| and |pitch| (ZYX convention) are
below the threshold (default 2°), otherwise an error: the frame is not gravity-aligned.
