# `manipulation/_geometry_kernels.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_plane_sums`

[定义位置](../../../manipulation/_geometry_kernels.py#L19)。

输入签名：

```python
def _plane_sums(low, high, normal):
```

实现说明：

> Keep x/y/z multiplication and addition order for one halfspace.
>
> Each row has exactly three coordinates. The original NumPy calculation
> reduces this length-three axis; the probe compares its four sums directly
> as well as the final classification. Error sums use the same selected
> products, and no fused multiply-add or reassociation is requested.

显式输出分支：

```python
(minimum, maximum, low_absolute, high_absolute)
```

读取的对象属性：

无显式条目。

写入目标：

```python
minimum
maximum
low_absolute
high_absolute
axis
coefficient
low_term
high_term
```

直接调用符号：

```python
range
abs
njit
```

## `classify_box`

[定义位置](../../../manipulation/_geometry_kernels.py#L47)。

输入签名：

```python
def classify_box(low, high, normals, bounds):
```

实现说明：

> Return 1=inside, -1=outside, 0=mixed with original precedence.
>
> A region is a conjunction of halfspaces. The caller supplies its complete
> finite rows; zero rows represent the whole space, matching NumPy all([]).

显式输出分支：

```python
1
-1
0
```

读取的对象属性：

```python
normals.shape
```

写入目标：

```python
inside
outside
plane
minimum
maximum
low_absolute
high_absolute
low_error
high_error
```

直接调用符号：

```python
range
_plane_sums
njit
```

## `norm3`

[定义位置](../../../manipulation/_geometry_kernels.py#L73)。

输入签名：

```python
def norm3(x, y, z):
```

实现说明：

> Use the original length-three sqrt(dot(v,v)) operation order.
>
> NumPy's one-dimensional norm takes this dot/sqrt path. Numba's generic
> norm instead calls a scaled BLAS nrm2, which is a different computation
> for underflow/overflow and can round differently at ordinary magnitudes.
> Explicit products and left-to-right additions implement that three-term
> formula. Boundary probes and actual-query comparisons establish equality
> for this environment; a different BLAS reduction is a distinct baseline.

显式输出分支：

```python
np.sqrt(x * x + y * y + z * z)
```

读取的对象属性：

```python
np.sqrt
```

写入目标：

无显式条目。

直接调用符号：

```python
np.sqrt
njit
```

## `box_distance`

[定义位置](../../../manipulation/_geometry_kernels.py#L87)。

输入签名：

```python
def box_distance(low_a, high_a, low_b, high_b):
```

实现说明：

> Lower-bound complete box separation with the original error allowance.

显式输出分支：

```python
max(0.0, np.nextafter(value - error, -np.inf))
```

读取的对象属性：

```python
np.nextafter
np.inf
```

写入目标：

```python
gx
gy
gz
value
magnitude
axis
error
```

直接调用符号：

```python
max
norm3
range
abs
np.nextafter
njit
```

## `projection_lower`

[定义位置](../../../manipulation/_geometry_kernels.py#L105)。

输入签名：

```python
def projection_lower(low, high, direction, native_low, native_high, length):
```

实现说明：

> Project a complete box using original xyz sums and outward rounding.

显式输出分支：

```python
max(0.0, np.nextafter((gap - error) / (length * (1.0 + _GAMMA)), -np.inf))
```

读取的对象属性：

```python
np.nextafter
np.inf
```

写入目标：

```python
minimum
maximum
low_absolute
high_absolute
box_low
box_high
gap
error
```

直接调用符号：

```python
_plane_sums
np.nextafter
max
abs
njit
```
