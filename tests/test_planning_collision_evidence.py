"""Same-query scene evidence keeps collision, coverage and conditional meanings."""

# BEGIN saved-curve scene classification uses the original component query.
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from manipulation.planning import _ProgressMap, _scene_boundary_evaluate


@pytest.mark.parametrize("category", ["scene_update/background/robot", "unsupported/contact/robot/owner",
    "unknown/forbidden/robot", "contact/required/rule", "gripper/controller",
    "conditional/collision/robot/background", "conditional/unknown/robot", "unknown/other/robot"])
def test_noncollision_components_do_not_generate_collision_evidence(category):
    calls = []
    def evaluate(spheres, p, r, s, q, names, *, components_out, **kwargs):
        calls.append(1)
        components_out[category] = torch.ones_like(s)
        return torch.zeros_like(s) if category.startswith("conditional/") else torch.ones_like(s)
    scene = SimpleNamespace(required_progress_anchors=(), evaluate=evaluate)
    s = torch.tensor([[0., .5, 1.]])
    evidence = {}
    result = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        torch.zeros(1, 3, 1, 4), torch.zeros(1, 3, 3), torch.zeros(1, 3, 4), s,
        torch.zeros(1, 3, 1), ("q",), trajectory_time=s, normalized_time=s, scene_evidence_out=evidence)
    assert calls == [1]
    assert bool((result == (0 if category.startswith("conditional/") else 1)).all())
    assert bool((evidence["collision"] == 0).all())
    # BEGIN numerical support is retained without becoming collision evidence.
    assert bool((evidence["scene_update"] == int(category.startswith("scene_update/"))).all())
    # BEGIN only the explicit forbidden-space component authorizes coverage search.
    assert bool((evidence["coverage"] == int(category.startswith("unknown/forbidden/"))).all())
    # BEGIN only conditional unknown enters the fourth evidence channel.
    assert bool((evidence["conditional_unknown"] == int(category.startswith("conditional/unknown/"))).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_collision_mode_max_does_not_follow_support_max_or_change_its_gradient(dtype):
    s = torch.tensor([[.5]], dtype=dtype)
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    calls = []
    def evaluate(spheres, p, r, progress, q, names, *, components_out=None, **kwargs):
        calls.append(progress.shape[1])
        collision = .25*q[..., 0].square()+2*(progress > .5)
        support = q[..., 0].square()+4*(progress < .5)
        if components_out is not None:
            components_out.update({"collision/robot/owner": collision, "scene_update/robot/owner": support,
                "conditional/collision/robot/owner": torch.full_like(progress, 100.)})
        return collision+support
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate)
    results = []
    for capture in (False, True):
        q = torch.ones(1, 1, 1, dtype=dtype, requires_grad=True)
        evidence = {} if capture else None
        result = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.), mapping,
            torch.zeros(1, 1, 1, 4, dtype=dtype), torch.zeros(1, 1, 3, dtype=dtype),
            torch.zeros(1, 1, 4, dtype=dtype), s, q, ("q",),
            trajectory_time=s*2, normalized_time=s, scene_evidence_out=evidence)
        results.append((result.detach(), torch.autograd.grad(result.sum(), q)[0]))
        if capture:
            assert evidence["collision"].item() == 2.25  # right-side collision, left-side hard maximum
            # BEGIN each diagnostic channel takes its own boundary-mode maximum.
            assert evidence["scene_update"].item() == 5.
            assert not evidence["scene_update"].requires_grad
            # END the left support witness cannot hide the right collision witness.
            assert not evidence["collision"].requires_grad
    assert calls == [3, 3], "Each evaluation queries exact/left/right geometry in one batch"
    assert results[0][0].item() == 5.25
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_scene_diagnostic_retains_nonfinite_witness_beside_finite_zero(nonfinite):
    def evaluate(spheres, p, r, s, q, names, *, components_out, **kwargs):
        components_out["scene_update/background/robot"] = s.new_tensor([[nonfinite, 0.]])
        return components_out["scene_update/background/robot"]
    scene = SimpleNamespace(required_progress_anchors=(), evaluate=evaluate)
    s = torch.tensor([[0., 1.]])
    diagnostics = {}
    _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), s, torch.zeros(1, 2, 1), ("q",),
        trajectory_time=s, normalized_time=s, component_diagnostics=diagnostics)
    assert not np.isfinite(diagnostics["scene_update/background/robot"]["maximum"])
    assert diagnostics["scene_update/background/robot"]["time_s"] == 0.


def test_coverage_mode_max_preserves_all_actor_terms_without_changing_hard_gradient():
    s = torch.tensor([[.5]], dtype=torch.float64)
    calls = []
    def evaluate(spheres, p, r, progress, q, names, *, components_out, **kwargs):
        calls.append(progress.shape[1])
        components_out.update({
            "collision/robot/owner": 2*(progress > .5).to(q.dtype),
            "scene_update/robot/owner": 5*(progress < .5).to(q.dtype),
            "unknown/forbidden/robot": q[..., 0].square()+3*(progress == .5),
            "unknown/forbidden/payload": .5*q[..., 0].square(),
            "conditional/unknown/robot": torch.full_like(progress, 100.)})
        return sum((value for name, value in components_out.items()
                    if not name.startswith("conditional/")), torch.zeros_like(progress))
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate)
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    q = torch.ones(1, 1, 1, dtype=torch.float64, requires_grad=True)
    evidence = {}
    result = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.), mapping,
        torch.zeros(1, 1, 1, 4, dtype=torch.float64), torch.zeros(1, 1, 3, dtype=torch.float64),
        torch.zeros(1, 1, 4, dtype=torch.float64), s, q, ("q",), trajectory_time=s,
        normalized_time=s, scene_evidence_out=evidence)
    assert calls == [3]
    assert evidence["coverage"].item() == 4.5  # exact mode; both robot and payload count
    assert evidence["scene_update"].item() == 5.  # left mode
    assert evidence["collision"].item() == 2.  # right mode
    # BEGIN permitted uncertainty is retained without hard cost or gradients.
    assert evidence["conditional_unknown"].item() == 100.
    assert not evidence["conditional_unknown"].requires_grad
    # END the hard result below is unchanged despite positive unknown evidence.
    assert result.item() == 6.5  # the original left-mode hard maximum
    assert torch.autograd.grad(result.sum(), q)[0].item() == 3.
    assert not evidence["coverage"].requires_grad


@pytest.mark.parametrize('hard', ['collision/robot/background','collision/robot/self',
    'collision/payload/background','scene_update/background/robot','unsupported/contact/robot/owner',
    'unknown/forbidden/robot','contact/required/rule','gripper/controller'])
def test_tiny_hard_residual_is_retained_beside_large_conditional_evidence(hard):
    calls=[]
    def evaluate(spheres,p,r,progress,q,names,*,components_out,**kwargs):
        calls.append(progress.clone())
        components_out[hard]=torch.full_like(progress,1e-30)
        components_out['conditional/unknown/robot']=torch.full_like(progress,1e20)
        return components_out[hard]
    scene=SimpleNamespace(required_progress_anchors=(),evaluate=evaluate)
    s=torch.tensor([[0.,1.]],dtype=torch.float64)
    evidence,diagnostics={},{}
    result=_scene_boundary_evaluate(scene,SimpleNamespace(start_progress=0.,end_progress=1.),
        _ProgressMap(np.array([0.,1.]),np.array([0.,1.])),torch.zeros(1,2,0,4),
        torch.zeros(1,2,3),torch.zeros(1,2,4),s,torch.zeros(1,2,1),('q',),
        trajectory_time=s,normalized_time=s,scene_evidence_out=evidence,component_diagnostics=diagnostics)
    assert len(calls)==1
    assert bool((result==1e-30).all())
    assert diagnostics[hard]['maximum']==1e-30
    assert diagnostics[hard]['time_s']==0.
    assert evidence['conditional_unknown'].tolist()==[[1e20,1e20]]


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_collision_retains_original_query_location(bad):
    def evaluate(spheres,p,r,progress,q,names,*,components_out,**kwargs):
        components_out['collision/robot/background']=progress.new_tensor([[0.,bad]])
        return components_out['collision/robot/background']
    scene=SimpleNamespace(required_progress_anchors=(),evaluate=evaluate)
    s=torch.tensor([[0.,1.]],dtype=torch.float64)
    evidence,diagnostics={},{}
    result=_scene_boundary_evaluate(scene,SimpleNamespace(start_progress=0.,end_progress=1.),
        _ProgressMap(np.array([0.,1.]),np.array([0.,1.])),torch.zeros(1,2,0,4),
        torch.zeros(1,2,3),torch.zeros(1,2,4),s,torch.zeros(1,2,1),('q',),
        trajectory_time=2*s,normalized_time=s,scene_evidence_out=evidence,component_diagnostics=diagnostics)
    assert not torch.isfinite(result[0,1])
    assert not torch.isfinite(evidence['collision'][0,1])
    assert diagnostics['collision/robot/background']['time_s']==2.
    assert diagnostics['collision/robot/background']['progress']==1.
# END category evidence and physical witness times remain independent of aggregate magnitude.
