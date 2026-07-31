# Retargeting Algorithm

[中文版](../../zh/reference/retargeting.md)

somehand solves each hand independently. Bi-hand retargeting runs the same single-hand pipeline once for the left hand and once for the right hand.

## Pipeline

```text
21 hand landmarks
        │
        ▼
wrist-centered palm frame + side transform
        │
        ▼
landmark temporal filter
        │
        ▼
vector, frame, and distance targets
        │
        ▼
bounded joint-angle optimization
        │
        ▼
joint-output smoothing + mimic joints
```

Landmarks are translated to make the wrist point the origin. Points `0`, `5`, and `9` define a palm frame; a side-specific transform maps that frame into robot coordinates. If the palm frame is degenerate, the implementation falls back to a fixed MediaPipe-to-MuJoCo axis mapping. An exponential moving average then smooths the transformed landmarks.

## Optimization Objective

Let `q` be the robot joint positions. MuJoCo forward kinematics provides robot body/site positions and rotations. The solver minimizes:

```text
loss(q) = direction terms
        + frame-axis terms
        + active distance terms
        + norm_delta * ||q - q_previous||²
```

### Vector Direction

A vector constraint maps a human landmark pair to a robot body/site pair. Both vectors are normalized, so the term measures direction rather than length:

```text
direction_loss = weight * (1 - dot(robot_direction, human_direction))
```

The loss is zero when the directions agree and increases as they diverge.

### Frame Orientation

A frame constraint uses an origin, primary point, and secondary point. Gram-Schmidt orthogonalization builds two human target axes. The corresponding local robot axes are derived from the model and rotated by the current origin body/site orientation. Primary and secondary axes each use the same cosine-direction loss with separate weights.

Use frame constraints when matching one vector is insufficient to determine rotation around that vector, such as thumb-base orientation.

### Fingertip Distance

A distance constraint first computes a target robot distance:

```text
raw:         target = scale * human_distance
hand_scaled: target = scale * human_distance
                      * robot_middle_finger_length
                      / human_middle_finger_length
```

The checked-in retargeting configs use `raw`, so the input landmark distance is applied without hand-size compensation. `hand_scaled` remains available when compensation is needed; the middle-finger length is the sum of the `9 → 10 → 11 → 12` human segments and the resolved robot middle-finger chain.

Distance activation depends on the human distance `d` and `threshold`:

```text
linear:   activation = max(0, 1 - d / threshold)
gaussian: activation = exp(-(d / (threshold / 2))²)
```

If `threshold <= 0`, activation is always `1`. Activation is temporally smoothed before use. The distance term is one-sided:

```text
distance_loss = weight * activation
                * max(robot_distance - target_distance, 0)²
```

It pulls robot points together when they are too far apart, but does not penalize them for being closer than the target. This makes the term suitable for pinch closure.

### Temporal Regularization

`norm_delta * ||q - q_previous||²` discourages large changes from the previous output. Despite its name, `norm_delta` is a loss weight, not a convergence tolerance.

## Solver

The optimizer is SciPy SLSQP with MuJoCo position/rotation Jacobians and an analytic objective gradient. It:

- starts from the current joint state;
- optimizes independent joints only and reconstructs mimic joints;
- enforces MJCF joint limits as bounds;
- stops after `max_iterations` or SLSQP convergence;
- uses an internal, fixed SLSQP `ftol` of `1e-6`.

After optimization, another exponential moving average blends the result with the previous output.

## Solver and Smoothing Parameters

All checked-in model configs extend `base/_universal_common.yaml`. “Built-in effective” is the value after inheritance and library fallback; “Fallback” is what a custom config gets when it does not provide or inherit the field.

| Parameter | Built-in effective | Fallback | Effect |
| --- | ---: | ---: | --- |
| `preprocess.temporal_filter_alpha` | `0.65` | `0.35` | Landmark EMA coefficient. Lower is smoother but adds input lag; higher is more responsive but passes more noise. Valid range: `(0, 1]`. |
| `solver.max_iterations` | `60` | `30` | SLSQP iteration cap per frame. Higher can improve difficult-frame convergence but costs CPU time. |
| `solver.norm_delta` | `0.001` | `0.01` | Previous-output penalty weight. Higher suppresses joint jumps but resists fast motion. Keep non-negative. |
| `solver.output_alpha` | `0.92` | `0.70` | Joint-output EMA coefficient. Lower is smoother but adds output lag; higher follows the optimizer more closely. Valid range: `(0, 1]`. |
| `solver.activation_alpha` | `0.30` | `0.30` | Distance-activation EMA coefficient. Lower makes pinch activation change more slowly. Keep in `(0, 1]`. |

For any EMA in this page:

```text
filtered_t = alpha * current_t + (1 - alpha) * filtered_(t-1)
```

The first frame uses the current value directly.

## Constraint Parameters

| Parameter | Effect |
| --- | --- |
| `weight` | Multiplies one vector or distance loss. A larger value gives that constraint more influence relative to the other terms. |
| `primary_weight`, `secondary_weight` | Independently weight the two axes of a frame constraint. |
| `scale` | Multiplies the human distance when producing the robot target distance. |
| `threshold` | Controls how far apart human points may be before a distance constraint deactivates. It uses the input landmark distance unit. A larger value activates pinch constraints earlier. |
| `activation_type` | Selects `linear` or `gaussian` distance activation. |
| `scale_mode` | `raw` (the default) uses the input distance directly; `hand_scaled` compensates for human/robot hand size. |

Values in an individual constraint override `constraint_defaults`. Vector `terminal_weight` applies when the second robot point is a site; distance `weights_by_human` selects defaults by landmark pair.

## Tuning Order

1. Run with `--viewer-mode diagnostic` and verify landmark pairs and robot body/site names first.
2. Tune `temporal_filter_alpha` and `output_alpha` for the noise/latency balance.
3. Tune `norm_delta` only if joint jumps remain or motion becomes too resistant.
4. Tune constraint weights, distance `scale`, and `threshold` for pose-specific errors.
5. Increase `max_iterations` only when the optimizer needs more work to reach a useful pose.

| Symptom | First parameters to inspect |
| --- | --- |
| Noisy landmarks or joint jitter | Lower `temporal_filter_alpha` or `output_alpha`; then consider increasing `norm_delta`. |
| Motion responds too slowly | Raise the two alpha values or reduce `norm_delta`. |
| Pinch closes too late or not enough | Check the fingertip sites, then raise `threshold`, `weight`, or adjust `scale`. |
| Pinch closes too early | Lower `threshold` or the distance `weight`. |
| Correct finger bend but wrong base orientation | Inspect the frame constraint and its two weights. |
| Per-frame solve is too slow | Lower `max_iterations`; profile before changing constraint coverage. |

Change one parameter group at a time and verify several representative poses. YAML structure and path rules are documented in [Configuration](configuration.md).
