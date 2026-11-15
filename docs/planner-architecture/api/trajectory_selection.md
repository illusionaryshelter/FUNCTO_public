# `manipulation/trajectory_selection.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../14-grounding-planning-portfolio.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `metric_values`

[定义位置](../../../manipulation/trajectory_selection.py#L11)。

输入签名：

```python
def metric_values(case):
```

实现说明：

> Expose the declared four metrics in cost units, metres and radians.

显式输出分支：

```python
dict(original_objective=case['original_objective'], minimum_joint_position_margin_rad=case['joint']['position']['minimum_margin']['margin'], minimum_required_robot_pair_clearance_m=case['robot_self']['minimum']['evidence']['signed_distance_m'], tool_rotation_length_rad=radians(case['tool_movement']['rotation_travel_deg']))
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
dict
radians
```

## `score_metrics`

[定义位置](../../../manipulation/trajectory_selection.py#L20)。

输入签名：

```python
def score_metrics(values, criteria):
```

实现说明：

> Return affine cost components; reference and swing share native units.
>
> Positive cost_sign denotes a smaller-is-better metric. Negative denotes
> a larger-is-better metric. Values outside the fixed calibration interval
> retain the same linear exchange rate. Scores are preference units.

显式输出分支：

```python
dict(score=fsum((row['weighted_cost'] for row in components)), components=components)
```

读取的对象属性：

```python
components.append
```

写入目标：

```python
components
criterion
value
normalized
row
```

直接调用符号：

```python
components.append
dict
fsum
```

## `select`

[定义位置](../../../manipulation/trajectory_selection.py#L37)。

输入签名：

```python
def select(comparison, calibration):
```

实现说明：

> Rank only the original accepted set and preserve every exact minimizer.
>
> An empty accepted set is a planning refusal. Multiple score minimizers
> remain an explicit user choice, following the requested pause boundary;
> display rounding does not create a numerical tie rule.

显式输出分支：

```python
dict(status='incomplete_metrics' if missing else 'selected' if len(best) == 1 else 'tie_requires_user_choice' if best else 'no_accepted_plan', eligible_candidate_ids=[row['id'] for row in comparison['cases'] if row['accepted']], scored_candidate_ids=[row['id'] for row in scored], scores=scored, rejected=rejected, accepted_with_missing_metrics=missing, best_candidate_ids=best, preferred_id=best[0] if len(best) == 1 else None, physical_status=comparison['physical_status'], calibration=calibration, score_direction='min', score_semantics='fixed affine preference units; not a probability')
```

读取的对象属性：

```python
case.get
missing.append
scored.append
rejected.append
```

写入目标：

```python
scored
rejected
missing
case
minimum
row
best
```

直接调用符号：

```python
case.get
missing.append
dict
scored.append
score_metrics
metric_values
rejected.append
min
len
```
