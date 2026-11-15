# `transfer/cross_instance/refinement.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../../../transfer/cross_instance/ARCHITECTURE.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_oareg_module`

[定义位置](../../../transfer/cross_instance/refinement.py#L32)。

输入签名：

```python
def _oareg_module(relative, name):
```

实现说明：

> Load the maintained workspace OAReg implementation by its exact file.
>
> OAReg's source repository is an existing dependency, rather than an
> installed Python namespace.  File loading follows FUNCTO's MARCO adapter
> and keeps its generic ``model`` package names out of the caller's import
> path.  The loaded numerical implementation is used unchanged.

显式输出分支：

```python
module
```

读取的对象属性：

```python
Path(__file__).resolve().parents
Path(__file__).resolve
importlib.util.spec_from_file_location
importlib.util
importlib.util.module_from_spec
spec.loader.exec_module
spec.loader
```

写入目标：

```python
repository
spec
module
```

直接调用符号：

```python
Path(__file__).resolve
Path
importlib.util.spec_from_file_location
importlib.util.module_from_spec
spec.loader.exec_module
```

## `NeuralRefinementFailure`

[定义位置](../../../transfer/cross_instance/refinement.py#L58)。

```python
class NeuralRefinementFailure(RuntimeError):
```

类字段：

无显式条目。

实现说明：

> Numerical failure carrying exact partial arrays and JSON-safe metadata.

## `NeuralRefinementFailure.__init__`

[定义位置](../../../transfer/cross_instance/refinement.py#L61)。

输入签名：

```python
def __init__(self, message, arrays, report):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
```

写入目标：

```python
self.arrays
self.report
```

直接调用符号：

```python
super().__init__
super
```

## `prepare_refinement`

[定义位置](../../../transfer/cross_instance/refinement.py#L67)。

输入签名：

```python
def prepare_refinement(source, target, appearance_cost, source_owner, target_owner, initial_mapped, source_frame, initial_rotation, *, n_neighbors=30):
```

实现说明：

> Build fixed units, source graph and the actual two neural coordinate axes.
>
> The source graph is the existing OAReg barycentric graph, evaluated once
> on all source rows. Its signed weights enter LLR unchanged. Every positive
> geometric neighbor edge enters the strain mean once in its directed graph
> occurrence; zero-length source edges have no ratio and are omitted from
> that edge list, while their source identities and LLR rows are retained.
> The returned arrays/tensors are also the independently testable loss API.

显式输出分支：

```python
dict(source=source, target=target, source_owner=np.asarray(source_owner, dtype=float), target_owner=np.asarray(target_owner, dtype=float), source_center=source_center, source_rms=length, target_origin=origin, source_frame=np.asarray(source_frame, dtype=float), initial_rotation=np.asarray(initial_rotation, dtype=float), arrays=numpy_values, tensors=tensors, n_neighbors=n_neighbors, omitted_zero_length_edges=int(np.count_nonzero(~positive)))
```

读取的对象属性：

```python
np.asarray
source.mean
np.sqrt
np.mean
np.sum
initial_mapped.mean
graph.indices.reshape
graph.indices
graph.data.reshape
graph.data
np.repeat
np.arange
neighbors.ravel
owner.mean
torch.as_tensor
torch.long
torch.float64
numpy_values.items
np.count_nonzero
```

写入目标：

```python
source
target
initial_mapped
source_center
length
origin
x
y
z0
source_axes
output_axes
graph
neighbors
weights
edge_source
edge_target
edge_length_squared
positive
owner
owner_mean
owner_variance
numpy_values
integer_keys
tensors
name
value
```

直接调用符号：

```python
np.asarray
source.mean
float
np.sqrt
np.mean
np.sum
initial_mapped.mean
barycenter_kneighbors_graph
graph.indices.reshape
len
graph.data.reshape
np.repeat
np.arange
neighbors.ravel
owner.mean
dict
context_vector
torch.as_tensor
numpy_values.items
int
np.count_nonzero
```

## `objective_components`

[定义位置](../../../transfer/cross_instance/refinement.py#L122)。

输入签名：

```python
def objective_components(mapped_normalized, columns, rotation, prepared):
```

实现说明：

> Return the seven differentiable terms for fixed J and relation rotation.
>
> Z, Y and owner moments use the common source-RMS unit. q is computed from
> the SAME Z, including its centre and RMS. Thus the relation rotation is
> tied to the predicted shape. No derivative through SVD is required in
> this fixed-R block. The final component is the sum of all six unit-weight
> terms; the appearance term is constant only during this fixed-J block.

显式输出分支：

```python
torch.cat((terms, terms.sum()[None]))
```

读取的对象属性：

```python
torch.as_tensor
torch.long
z.dtype
z.mean
torch.sqrt
torch.mean
torch.sum
centered.square
offset.square
edges.square
torch.stack
t['appearance'][torch.arange(len(z)), columns].mean
torch.arange
torch.sum((z - t['y'][columns]).square(), dim=1).mean
(z - t['y'][columns]).square
torch.linalg.vector_norm(reconstruction, dim=1).mean
torch.linalg.vector_norm
torch.linalg
(ratio_squared - 1.0 - torch.log(ratio_squared)).mean
torch.log
torch.sum((t['x'] @ rotation.T - q).square(), dim=1).mean
(t['x'] @ rotation.T - q).square
rotation.T
(rotation @ t['source_context'] - context).square
torch.cat
terms.sum
```

写入目标：

```python
t
z
columns
rotation
center
centered
rms
q
offset
context
reconstruction
edges
ratio_squared
terms
```

直接调用符号：

```python
torch.as_tensor
z.mean
torch.sqrt
torch.mean
torch.sum
centered.square
offset.square
edges.square
torch.stack
t['appearance'][torch.arange(len(z)), columns].mean
torch.arange
len
torch.sum((z - t['y'][columns]).square(), dim=1).mean
(z - t['y'][columns]).square
torch.linalg.vector_norm(reconstruction, dim=1).mean
torch.linalg.vector_norm
(ratio_squared - 1.0 - torch.log(ratio_squared)).mean
torch.log
torch.sum((t['x'] @ rotation.T - q).square(), dim=1).mean
(t['x'] @ rotation.T - q).square
(rotation @ t['source_context'] - context).square
torch.cat
terms.sum
```

## `assign_full_target`

[定义位置](../../../transfer/cross_instance/refinement.py#L155)。

输入签名：

```python
def assign_full_target(mapped_normalized, prepared):
```

实现说明：

> Exactly minimize J over every actual target element, allowing reuse.

显式输出分支：

```python
(a['appearance'] + cdist(np.asarray(mapped_normalized), a['y'], 'sqeuclidean')).argmin(axis=1)
```

读取的对象属性：

```python
(a['appearance'] + cdist(np.asarray(mapped_normalized), a['y'], 'sqeuclidean')).argmin
np.asarray
```

写入目标：

```python
a
```

直接调用符号：

```python
(a['appearance'] + cdist(np.asarray(mapped_normalized), a['y'], 'sqeuclidean')).argmin
cdist
np.asarray
```

## `relation_rotation`

[定义位置](../../../transfer/cross_instance/refinement.py#L162)。

输入签名：

```python
def relation_rotation(mapped_normalized, prepared):
```

实现说明：

> Use the existing exact proper-SVD block for this predicted shape.

显式输出分支：

```python
(evidence['rotation_wahba'], evidence)
```

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
world
evidence
```

直接调用符号：

```python
np.asarray
orientation_evidence
```

## `refine_identity_mapping`

[定义位置](../../../transfer/cross_instance/refinement.py#L170)。

输入签名：

```python
def refine_identity_mapping(source, target, appearance_cost, source_owner, target_owner, initial_mapped, source_ids, target_ids, source_frame, initial_rotation, *, steps=200, n_neighbors=30, seed=0):
```

实现说明：

> Fit OAReg's residual SIREN with complete C/Y/source identities.
>
> Each outer block uses fresh torch.optim.LBFGS(max_iter=20,
> line_search_fn='strong_wolfe'), with its curvature history retained inside
> this fixed-J/R block. The actual final loss is recomputed because step()
> returns its starting loss. Exact R and full-domain J updates follow it.
>
> The final linear layer is zeroed so the initial map is precisely the
> supplied coarse map. Source functional axes parameterize network inputs;
> the coarse-transported axes parameterize displacements. Independent rigid
> changes of the two world frames and common unit changes preserve these
> local neural coordinates as well as the objective.
>
> All accepted block states and every closure trial are returned. A
> nonfinite objective/parameter gradient terminates with partial evidence;
> no epsilon, clamp, optimizer substitution or task-specific branch alters
> the mathematical domain. Finite nonmonotone steps remain in the record.

显式输出分支：

```python
finish('computed_neural_refinement')
```

读取的对象属性：

```python
torch.manual_seed
Siren(in_features=3, hidden_features=128, hidden_layers=3, out_features=3, outermost_linear=True, first_omega_0=30, hidden_omega_0=30.0).to
torch.float64
torch.no_grad
model.net[-1].weight.zero_
model.net[-1].weight
model.net
model.net[-1].bias.zero_
model.net[-1].bias
states.append
states[-1]['energy_components'].copy
columns.copy
rotation.copy
torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone
torch.cat
parameter.detach().reshape
parameter.detach
model.parameters
torch.optim.LBFGS
torch.optim
optimizer.step
prediction().numpy
objective_components(torch.as_tensor(z), fit_columns, fit_rotation, p).numpy
torch.as_tensor
objective_components(torch.as_tensor(z), fit_columns, rotation, p).detach().numpy
objective_components(torch.as_tensor(z), fit_columns, rotation, p).detach
optimizer.state
blocks.append
returned_initial.detach
np.nan
torch.dot
torch.linalg.vector_norm
torch.linalg
torch.max
torch.abs
```

写入目标：

```python
started
p
t
model
states
blocks
trials
columns
rotation
evidence
step
before
fit_columns
fit_rotation
parameters_before
parameter
optimizer
returned_initial
z
after_fit
after_rotation
final
optimizer_state
search_started
parameter_step
```

直接调用符号：

```python
perf_counter
prepare_refinement
torch.manual_seed
Siren(in_features=3, hidden_features=128, hidden_layers=3, out_features=3, outermost_linear=True, first_omega_0=30, hidden_omega_0=30.0).to
Siren
torch.no_grad
model.net[-1].weight.zero_
model.net[-1].bias.zero_
assign_full_target
relation_rotation
states.append
state
range
states[-1]['energy_components'].copy
columns.copy
rotation.copy
torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone
torch.cat
parameter.detach().reshape
parameter.detach
model.parameters
torch.optim.LBFGS
optimizer.step
prediction().numpy
prediction
objective_components(torch.as_tensor(z), fit_columns, fit_rotation, p).numpy
objective_components
torch.as_tensor
objective_components(torch.as_tensor(z), fit_columns, rotation, p).detach().numpy
objective_components(torch.as_tensor(z), fit_columns, rotation, p).detach
next
iter
blocks.append
dict
float
returned_initial.detach
len
torch.dot
torch.linalg.vector_norm
torch.max
torch.abs
finish
```

## `refine_identity_mapping.prediction`

[定义位置](../../../transfer/cross_instance/refinement.py#L205)。

输入签名：

```python
def prediction():
```

显式输出分支：

```python
t['z0'] + model(t['model_input']) @ t['output_axes'].T
```

读取的对象属性：

```python
t['output_axes'].T
```

写入目标：

无显式条目。

直接调用符号：

```python
model
```

## `refine_identity_mapping.state`

[定义位置](../../../transfer/cross_instance/refinement.py#L212)。

输入签名：

```python
def state(z, j, r, relation):
```

显式输出分支：

```python
dict(mapped_points_world_m=p['target_origin'] + p['source_rms'] * z, target_indices=np.asarray(j).copy(), rotation_relation=np.asarray(r).copy(), energy_components=objective_components(torch.as_tensor(z), j, r, p).detach().numpy(), parameter_gradient_norm=float(torch.linalg.vector_norm(parameter_gradient).detach()), llr_residual_norms_normalized=torch.linalg.vector_norm(reconstruction, dim=1).numpy(), rotation_unique=relation['rotation_unique'], rotation_covariance_singular_values=relation['covariance_singular_values'])
```

读取的对象属性：

```python
model.zero_grad
current_terms[-1].backward
torch.cat
parameter.grad.reshape
parameter.grad
model.parameters
torch.as_tensor
torch.sum
np.asarray(j).copy
np.asarray
np.asarray(r).copy
objective_components(torch.as_tensor(z), j, r, p).detach().numpy
objective_components(torch.as_tensor(z), j, r, p).detach
torch.linalg.vector_norm(parameter_gradient).detach
torch.linalg.vector_norm
torch.linalg
torch.linalg.vector_norm(reconstruction, dim=1).numpy
```

写入目标：

```python
current_terms
parameter_gradient
parameter
reconstruction
```

直接调用符号：

```python
model.zero_grad
objective_components
prediction
current_terms[-1].backward
torch.cat
parameter.grad.reshape
model.parameters
torch.as_tensor
torch.sum
dict
np.asarray(j).copy
np.asarray
np.asarray(r).copy
objective_components(torch.as_tensor(z), j, r, p).detach().numpy
objective_components(torch.as_tensor(z), j, r, p).detach
float
torch.linalg.vector_norm(parameter_gradient).detach
torch.linalg.vector_norm
torch.linalg.vector_norm(reconstruction, dim=1).numpy
```

## `refine_identity_mapping.finish`

[定义位置](../../../transfer/cross_instance/refinement.py#L233)。

输入签名：

```python
def finish(status, failure=None):
```

实现说明：

> Package exact arrays; nonfinite metadata uses null, with raw NPZ kept.

显式输出分支：

```python
(arrays, report)
```

读取的对象属性：

```python
np.asarray(source_ids).copy
np.asarray
np.asarray(target_ids).copy
arrays.update
model.state_dict().items
model.state_dict
value.detach().numpy().copy
value.detach().numpy
value.detach
np.isfinite
p['target_origin'].tolist
np.max
```

写入目标：

```python
arrays
prefix
records
row
key
name
value
arrays['refinement_model_parameter__' + name]
final
arrays['refinement_failure_mapped_points_world_m']
final_terms
v
report
before
after
after_r
after_j
report['maximum_observed_block_increases']
```

直接调用符号：

```python
dict
np.asarray(source_ids).copy
np.asarray
np.asarray(target_ids).copy
arrays.update
model.state_dict().items
model.state_dict
value.detach().numpy().copy
value.detach().numpy
value.detach
np.isfinite
float
len
p['target_origin'].tolist
perf_counter
np.max
```

## `refine_identity_mapping.closure`

[定义位置](../../../transfer/cross_instance/refinement.py#L322)。

输入签名：

```python
def closure():
```

显式输出分支：

```python
values[-1]
```

读取的对象属性：

```python
optimizer.zero_grad
values.detach().numpy().copy
values.detach().numpy
values.detach
torch.isfinite(values).all
torch.isfinite
np.nan
values[-1].backward
torch.cat
parameter.grad.reshape
parameter.grad
model.parameters
torch.isfinite(flat_gradient).all
torch.linalg.vector_norm(flat_gradient).detach
torch.linalg.vector_norm
torch.linalg
trials.append
z.detach().numpy
z.detach
fit_columns.copy
fit_rotation.copy
np.isfinite
```

写入目标：

```python
z
values
values_np
finite_loss
gradient_norm
finite_gradient
flat_gradient
parameter
kind
failure
name
value
arrays
report
```

直接调用符号：

```python
optimizer.zero_grad
prediction
objective_components
values.detach().numpy().copy
values.detach().numpy
values.detach
bool
torch.isfinite(values).all
torch.isfinite
values[-1].backward
torch.cat
parameter.grad.reshape
model.parameters
torch.isfinite(flat_gradient).all
float
torch.linalg.vector_norm(flat_gradient).detach
torch.linalg.vector_norm
trials.append
dict
z.detach().numpy
z.detach
fit_columns.copy
fit_rotation.copy
len
zip
np.isfinite
finish
NeuralRefinementFailure
```
