# `transfer/cross_instance/coarse.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../../../transfer/cross_instance/ARCHITECTURE.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `owner_moments`

[定义位置](../../../transfer/cross_instance/coarse.py#L25)。

输入签名：

```python
def owner_moments(owner):
```

实现说明：

> Return the fixed observed mean and empirical squared radius in m².

显式输出分支：

```python
(mean, float(np.mean(np.sum(centered * centered, axis=1))))
```

读取的对象属性：

```python
np.mean
np.sum
```

写入目标：

```python
mean
centered
```

直接调用符号：

```python
_center
float
np.mean
np.sum
```

## `context_at`

[定义位置](../../../transfer/cross_instance/coarse.py#L31)。

输入签名：

```python
def context_at(center, owner_mean, owner_variance):
```

实现说明：

> Return d(c) and its 3x3 derivative with respect to metric centre c.
>
> With u=c-mu and rho²=variance+uᵀu, d=u/rho. The derivative has units
> 1/metre; the normalized translation derivative also multiplies source RMS.
> At c=mu the vector is zero while the derivative remains I/sqrt(variance).

显式输出分支：

```python
(u / rho, np.eye(3) / rho - np.outer(u, u) / rho ** 3)
```

读取的对象属性：

```python
np.asarray
np.sqrt
np.eye
np.outer
```

写入目标：

```python
u
rho
```

直接调用符号：

```python
np.asarray
float
np.sqrt
np.eye
np.outer
```

## `joint_residual`

[定义位置](../../../transfer/cross_instance/coarse.py#L43)。

输入签名：

```python
def joint_residual(parameters, x, paired_y, source_context, owner_mean, owner_variance, origin, length, initial_rotation, initial_translation):
```

实现说明：

> Six-variable residual for fixed source-to-target identities.
>
> x is centred and normalized by the source-region RMS. R=Exp(w)R_initial
> and t=t_initial+parameters[3:] give mapped world points origin+length*(R*x+t).
> Their centre is origin+length*t. Geometry is divided by sqrt(N), so the
> squared residual norm is mean geometry plus one unit-weight context term.
> SciPy minimizes half this norm; appearance is constant in this pose block.

显式输出分支：

```python
np.concatenate((geometry.ravel(), rotation @ source_context - target_context))
```

读取的对象属性：

```python
Rotation.from_rotvec(parameters[:3]).as_matrix
Rotation.from_rotvec
rotation.T
np.sqrt
np.concatenate
geometry.ravel
```

写入目标：

```python
rotation
translation
target_context
_
geometry
```

直接调用符号：

```python
Rotation.from_rotvec(parameters[:3]).as_matrix
Rotation.from_rotvec
context_at
np.sqrt
len
np.concatenate
geometry.ravel
```

## `fit_candidate_context`

[定义位置](../../../transfer/cross_instance/coarse.py#L60)。

输入签名：

```python
def fit_candidate_context(x, paired_y, source_context, owner_mean, owner_variance, origin, length, rotation, translation):
```

实现说明：

> Locally fit R/t with the existing SciPy solver and numerical Jacobian.
>
> The returned OptimizeResult retains termination status, evaluations and
> gradient. A successful local termination does not certify global optimum,
> correct correspondences, or physical pose accuracy. Numerical tolerances
> set optimizer precision, rather than task acceptance rules.

显式输出分支：

```python
(fitted_rotation, translation + result.x[3:], result)
```

读取的对象属性：

```python
np.zeros
Rotation.from_rotvec(result.x[:3]).as_matrix
Rotation.from_rotvec
result.x
```

写入目标：

```python
parameters
result
fitted_rotation
```

直接调用符号：

```python
np.zeros
least_squares
Rotation.from_rotvec(result.x[:3]).as_matrix
Rotation.from_rotvec
```

## `objective_terms`

[定义位置](../../../transfer/cross_instance/coarse.py#L83)。

输入签名：

```python
def objective_terms(x, y, appearance, columns, rotation, translation, source_context, owner_mean, owner_variance, origin, length):
```

实现说明：

> Appearance, normalized squared distance, current context, and total.

显式输出分支：

```python
np.concatenate((terms, [terms.sum()]))
```

读取的对象属性：

```python
rotation.T
np.array
appearance[np.arange(len(x)), columns].mean
np.arange
np.mean
np.sum
np.concatenate
terms.sum
```

写入目标：

```python
mapped
target_context
_
terms
```

直接调用符号：

```python
context_at
np.array
appearance[np.arange(len(x)), columns].mean
np.arange
len
np.mean
np.sum
np.concatenate
terms.sum
```

## `register_candidate_context`

[定义位置](../../../transfer/cross_instance/coarse.py#L96)。

输入签名：

```python
def register_candidate_context(source, target, raw, source_frame, source_owner, target_owner, *, steps=40, channels=1024):
```

实现说明：

> Fit every original region identity against the full observed domain.
>
> The 25 initial correspondence sets retain the existing raw/rotation-grid
> construction. Each begins from a fixed-context SVD evaluated at its paired
> centre. Thereafter every R/t and assignment block evaluates one shared
> moving-centre objective. At fixed pose context is independent of assignment,
> so the complete target-domain row argmin remains exact.
>
> Outputs retain all states and solver diagnostics. No solver-status-based
> candidate rejection or alternate-method branch is introduced. The caller
> decides how these numerical predictions are inspected and consumed.

显式输出分支：

```python
(arrays, report)
```

读取的对象属性：

```python
np.asarray
source.mean
np.sqrt
np.mean
np.sum
raw.argmax
target[raw_columns].mean
raw.max
Rotation.create_group('O').as_matrix
Rotation.create_group
basis.T
seeds.append
(appearance + cdist(x @ rotation.T, y, 'sqeuclidean')).argmin
rotation.T
names.append
seed.copy
y[columns].mean
rotation.copy
translation.copy
(appearance + cdist(normalized_map, y, 'sqeuclidean')).argmin
history.append
optimizer.x
optimizer.status
optimizer.success
optimizer.message
optimizer.optimality
optimizer.grad
optimizer.cost
optimizer.nfev
optimizer.njev
histories.append
arrays.update
arrays['coarse_energy_components'][:, -1, -1].argmin
```

写入目标：

```python
source
target
raw
center
length
x
raw_columns
origin
y
source_context
mean
variance
appearance
raw_rotation
basis
seeds
names
index
element
rotation
histories
seed
columns
seed_context
initial
translation
history
_
before
start_rotation
start_translation
optimizer
fitted
normalized_map
fresh
assigned
current_center
arrays
state
key
selected
report
```

直接调用符号：

```python
np.asarray
source.mean
float
np.sqrt
np.mean
np.sum
raw.argmax
target[raw_columns].mean
context_vector
owner_moments
raw.max
fit_rigid_with_context
context_at
enumerate
Rotation.create_group('O').as_matrix
Rotation.create_group
seeds.append
(appearance + cdist(x @ rotation.T, y, 'sqeuclidean')).argmin
cdist
names.append
str
seed.copy
y[columns].mean
range
objective_terms
rotation.copy
translation.copy
fit_candidate_context
(appearance + cdist(normalized_map, y, 'sqeuclidean')).argmin
history.append
dict
histories.append
arrays.update
int
arrays['coarse_energy_components'][:, -1, -1].argmin
len
```
