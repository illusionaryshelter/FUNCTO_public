"""Identity-preserving neural refinement over observed target elements.

This component owns the objective and alternating updates. OAReg supplies
its maintained SIREN representation and source barycentric reconstruction
graph; PyTorch supplies derivatives and fixed-block L-BFGS optimization. Every
source row has mass 1/N, and different rows may use the same target element.
All position arrays share a length unit. The source RMS defines the common
computational unit; the source functional frame and coarse rotation define
the neural input/output axes. No target-cloud normalization is performed.

The mathematical domain has positive source RMS, positive source-owner RMS
about the source-region centre, positive target-owner variance, and at least
one positive-length edge in the source graph. Finite
strain requires positive mapped lengths on those edges. A finite objective
therefore excludes a constant map, but does not establish injectivity or
exclude folds. LLR row norms use PyTorch's zero subgradient at zero residual.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist
import torch

from .context import context_vector, orientation_evidence


def _oareg_module(relative, name):
    """Load the maintained workspace OAReg implementation by its exact file.

    OAReg's source repository is an existing dependency, rather than an
    installed Python namespace.  File loading follows FUNCTO's MARCO adapter
    and keeps its generic ``model`` package names out of the caller's import
    path.  The loaded numerical implementation is used unchanged.
    """
    repository = Path(__file__).resolve().parents[3] / 'OAReg'
    spec = importlib.util.spec_from_file_location(name, repository / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Siren = _oareg_module('model/model.py', '_functo_oareg_siren').Siren
barycenter_kneighbors_graph = _oareg_module(
    'utils2/LLR.py', '_functo_oareg_llr').barycenter_kneighbors_graph


# BEGIN one full-evidence objective and its recorded alternating neural fit.
ENERGY_TERMS = ["appearance", "normalized_squared_distance", "llr_mean",
                "log_length_strain_mean", "local_direction_mean",
                "owner_context_squared", "total"]


class NeuralRefinementFailure(RuntimeError):
    """Numerical failure carrying exact partial arrays and JSON-safe metadata."""

    def __init__(self, message, arrays, report):
        super().__init__(message)
        self.arrays = arrays
        self.report = report


def prepare_refinement(source, target, appearance_cost, source_owner,
                       target_owner, initial_mapped, source_frame,
                       initial_rotation, *, n_neighbors=30):
    """Build fixed units, source graph and the actual two neural coordinate axes.

    The source graph is the existing OAReg barycentric graph, evaluated once
    on all source rows. Its signed weights enter LLR unchanged. Every positive
    geometric neighbor edge enters the strain mean once in its directed graph
    occurrence; zero-length source edges have no ratio and are omitted from
    that edge list, while their source identities and LLR rows are retained.
    The returned arrays/tensors are also the independently testable loss API.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    initial_mapped = np.asarray(initial_mapped, dtype=float)
    source_center = source.mean(axis=0)
    length = float(np.sqrt(np.mean(np.sum((source-source_center)**2, axis=1))))
    origin = initial_mapped.mean(axis=0)
    x = (source-source_center)/length
    y = (target-origin)/length
    z0 = (initial_mapped-origin)/length
    source_axes = np.asarray(source_frame, dtype=float)[:3, :3]
    output_axes = np.asarray(initial_rotation, dtype=float) @ source_axes
    graph = barycenter_kneighbors_graph(x, n_neighbors=n_neighbors, reg=1e-3)
    neighbors = graph.indices.reshape(len(x), n_neighbors)
    weights = graph.data.reshape(len(x), n_neighbors)
    edge_source = np.repeat(np.arange(len(x)), n_neighbors)
    edge_target = neighbors.ravel()
    edge_length_squared = np.sum((x[edge_source]-x[edge_target])**2, axis=1)
    positive = edge_length_squared > 0.
    owner = (np.asarray(target_owner, dtype=float)-origin)/length
    owner_mean = owner.mean(axis=0)
    owner_variance = float(np.mean(np.sum((owner-owner_mean)**2, axis=1)))
    numpy_values = dict(x=x, y=y, z0=z0,
        appearance=np.asarray(appearance_cost, dtype=float),
        source_context=context_vector(source_owner, source_center)["vector"],
        owner_mean=owner_mean, owner_variance=np.asarray(owner_variance),
        neighbors=neighbors, weights=weights,
        edge_source=edge_source[positive], edge_target=edge_target[positive],
        edge_length_squared=edge_length_squared[positive],
        model_input=x @ source_axes, output_axes=output_axes)
    integer_keys = {"neighbors", "edge_source", "edge_target"}
    tensors = {name: torch.as_tensor(value, dtype=(torch.long if name in integer_keys
                                                  else torch.float64))
               for name, value in numpy_values.items()}
    return dict(source=source, target=target,
        source_owner=np.asarray(source_owner, dtype=float),
        target_owner=np.asarray(target_owner, dtype=float),
        source_center=source_center, source_rms=length, target_origin=origin,
        source_frame=np.asarray(source_frame, dtype=float),
        initial_rotation=np.asarray(initial_rotation, dtype=float),
        arrays=numpy_values, tensors=tensors, n_neighbors=n_neighbors,
        omitted_zero_length_edges=int(np.count_nonzero(~positive)))


def objective_components(mapped_normalized, columns, rotation, prepared):
    """Return the seven differentiable terms for fixed J and relation rotation.

    Z, Y and owner moments use the common source-RMS unit. q is computed from
    the SAME Z, including its centre and RMS. Thus the relation rotation is
    tied to the predicted shape. No derivative through SVD is required in
    this fixed-R block. The final component is the sum of all six unit-weight
    terms; the appearance term is constant only during this fixed-J block.
    """
    t = prepared["tensors"]
    z = mapped_normalized
    columns = torch.as_tensor(columns, dtype=torch.long)
    rotation = torch.as_tensor(rotation, dtype=z.dtype)
    center = z.mean(dim=0)
    centered = z-center
    rms = torch.sqrt(torch.mean(torch.sum(centered.square(), dim=1)))
    q = centered/rms
    offset = center-t["owner_mean"]
    context = offset/torch.sqrt(t["owner_variance"]+torch.sum(offset.square()))
    reconstruction = z-torch.sum(t["weights"][:, :, None]*z[t["neighbors"]], dim=1)
    edges = z[t["edge_source"]]-z[t["edge_target"]]
    ratio_squared = torch.sum(edges.square(), dim=1)/t["edge_length_squared"]
    terms = torch.stack((
        t["appearance"][torch.arange(len(z)), columns].mean(),
        torch.sum((z-t["y"][columns]).square(), dim=1).mean(),
        torch.linalg.vector_norm(reconstruction, dim=1).mean(),
        (ratio_squared-1.-torch.log(ratio_squared)).mean(),
        torch.sum((t["x"] @ rotation.T-q).square(), dim=1).mean(),
        torch.sum((rotation @ t["source_context"]-context).square()),
    ))
    return torch.cat((terms, terms.sum()[None]))


def assign_full_target(mapped_normalized, prepared):
    """Exactly minimize J over every actual target element, allowing reuse."""
    a = prepared["arrays"]
    return (a["appearance"]+cdist(np.asarray(mapped_normalized), a["y"],
                                  "sqeuclidean")).argmin(axis=1)


def relation_rotation(mapped_normalized, prepared):
    """Use the existing exact proper-SVD block for this predicted shape."""
    world = prepared["target_origin"]+prepared["source_rms"]*np.asarray(mapped_normalized)
    evidence = orientation_evidence(prepared["source"], world,
        prepared["source_owner"], prepared["target_owner"])
    return evidence["rotation_wahba"], evidence


def refine_identity_mapping(source, target, appearance_cost, source_owner,
                            target_owner, initial_mapped, source_ids, target_ids,
                            source_frame, initial_rotation, *, steps=200,
                            n_neighbors=30, seed=0):
    """Fit OAReg's residual SIREN with complete C/Y/source identities.

    Each outer block uses fresh torch.optim.LBFGS(max_iter=20,
    line_search_fn='strong_wolfe'), with its curvature history retained inside
    this fixed-J/R block. The actual final loss is recomputed because step()
    returns its starting loss. Exact R and full-domain J updates follow it.

    The final linear layer is zeroed so the initial map is precisely the
    supplied coarse map. Source functional axes parameterize network inputs;
    the coarse-transported axes parameterize displacements. Independent rigid
    changes of the two world frames and common unit changes preserve these
    local neural coordinates as well as the objective.

    All accepted block states and every closure trial are returned. A
    nonfinite objective/parameter gradient terminates with partial evidence;
    no epsilon, clamp, optimizer substitution or task-specific branch alters
    the mathematical domain. Finite nonmonotone steps remain in the record.
    """
    started = perf_counter()
    p = prepare_refinement(source, target, appearance_cost, source_owner,
        target_owner, initial_mapped, source_frame, initial_rotation,
        n_neighbors=n_neighbors)
    t = p["tensors"]
    torch.manual_seed(seed)
    model = Siren(in_features=3, hidden_features=128, hidden_layers=3,
        out_features=3, outermost_linear=True, first_omega_0=30,
        hidden_omega_0=30.).to(dtype=torch.float64, device="cpu")
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.zero_()

    def prediction():
        return t["z0"]+model(t["model_input"]) @ t["output_axes"].T

    states, blocks, trials = [], [], []
    columns = assign_full_target(p["arrays"]["z0"], p)
    rotation, evidence = relation_rotation(p["arrays"]["z0"], p)

    def state(z, j, r, relation):
        # Evaluate the actual complete-state parameter gradient after exact
        # R/J updates. It is an optimization diagnostic, not an acceptance
        # condition; it need not equal a fixed-block closure-trial gradient.
        model.zero_grad(set_to_none=True)
        current_terms = objective_components(prediction(), j, r, p)
        current_terms[-1].backward()
        parameter_gradient = torch.cat([parameter.grad.reshape(-1)
            for parameter in model.parameters() if parameter.grad is not None])
        reconstruction = (torch.as_tensor(z)-torch.sum(t["weights"][:, :, None]
            *torch.as_tensor(z)[t["neighbors"]], dim=1))
        return dict(mapped_points_world_m=p["target_origin"]+p["source_rms"]*z,
            target_indices=np.asarray(j).copy(), rotation_relation=np.asarray(r).copy(),
            energy_components=objective_components(torch.as_tensor(z), j, r, p).detach().numpy(),
            parameter_gradient_norm=float(torch.linalg.vector_norm(parameter_gradient).detach()),
            llr_residual_norms_normalized=torch.linalg.vector_norm(reconstruction, dim=1).numpy(),
            rotation_unique=relation["rotation_unique"],
            rotation_covariance_singular_values=relation["covariance_singular_values"])

    states.append(state(p["arrays"]["z0"], columns, rotation, evidence))

    def finish(status, failure=None):
        """Package exact arrays; nonfinite metadata uses null, with raw NPZ kept."""
        arrays = dict(source_row_ids=np.asarray(source_ids).copy(),
            target_element_ids=np.asarray(target_ids).copy(),
            source_points_world_m=p["source"], target_points_world_m=p["target"],
            source_owner_points_world_m=p["source_owner"],
            target_owner_points_world_m=p["target_owner"],
            appearance_cost=p["arrays"]["appearance"],
            source_frame_world=p["source_frame"],
            refinement_initial_rotation=p["initial_rotation"],
            refinement_source_center_world_m=p["source_center"],
            refinement_source_rms_m=np.asarray(p["source_rms"]),
            refinement_target_origin_world_m=p["target_origin"],
            refinement_model_input=p["arrays"]["model_input"],
            refinement_output_axes=p["arrays"]["output_axes"],
            refinement_neighbor_indices=p["arrays"]["neighbors"],
            refinement_neighbor_weights=p["arrays"]["weights"],
            refinement_edge_source=p["arrays"]["edge_source"],
            refinement_edge_target=p["arrays"]["edge_target"],
            refinement_source_edge_length_squared_normalized=p["arrays"]["edge_length_squared"])
        for prefix, records in (("refinement_", states),
                                ("refinement_block_", blocks),
                                ("refinement_trial_", trials)):
            if records:
                arrays.update({prefix+key: np.asarray([row[key] for row in records])
                               for key in records[0]})
        for name, value in model.state_dict().items():
            arrays["refinement_model_parameter__"+name] = value.detach().numpy().copy()
        final = states[-1]
        arrays.update(mapped_points_world_m=final["mapped_points_world_m"],
            target_indices=final["target_indices"], rotation_relation=final["rotation_relation"])
        if failure is not None:
            arrays["refinement_failure_mapped_points_world_m"] = (
                trials[-1]["mapped_points_world_m"] if trials else final["mapped_points_world_m"])
        final_terms = [float(v) if np.isfinite(v) else None for v in final["energy_components"]]
        report = dict(status=status, method="full_evidence_siren_identity_refinement",
            source_rows=len(source), target_elements=len(target), source_identity_weight=1./len(source),
            target_reuse="permitted; no per-pixel material capacity",
            source_rms_m=p["source_rms"], target_origin_world_m=p["target_origin"].tolist(),
            energy_terms=ENERGY_TERMS, energy_weights=[1.]*6,
            initial_energy_components=[float(v) if np.isfinite(v) else None
                                       for v in states[0]["energy_components"]],
            final_energy_components=final_terms, requested_steps=steps,
            completed_steps=len(blocks), closure_evaluations=len(trials), seed=seed,
            n_neighbors=n_neighbors, positive_source_edges=len(p["arrays"]["edge_source"]),
            omitted_zero_length_source_edges=p["omitted_zero_length_edges"],
            optimizer=dict(api="torch.optim.LBFGS", max_iter=20, max_eval=25,
                line_search_fn="strong_wolfe", lr=1., tolerance_grad=1e-7,
                tolerance_change=1e-9, history_size=100,
                restart="each fixed-J/R block; curvature history retained only inside that block",
                step_return="initial objective; actual post-step terms recomputed"),
            representation=dict(api="OAReg.model.model.Siren", hidden_features=128,
                hidden_layers=3, dtype="float64", device="cpu", omega=30.,
                initial_output="zero last affine layer, exact supplied coarse map",
                input_axes="source functional frame", output_axes="coarse rotation times source functional axes"),
            # BEGIN distinguish recorded map evidence from the parameter trial.
            # A closure failure happens before its block can append a state.
            # The top-level map/energy therefore names the last recorded
            # complete block (or the initial state when no block completed),
            # while the serialized model parameters remain at the failed
            # line-search trial. Neither is substituted for the other.
            mapped_state=("final_completed_state" if failure is None else
                          "last_completed_state" if blocks else "initial_state"),
            mapped_state_index=len(states)-1,
            model_parameter_state=("failed_trial" if failure is not None else "final_completed_state"),
            model_parameter_trial_index=(failure["trial_index"] if failure is not None else None),
            # END the state labels also describe a failure before block one.
            final_parameter_gradient_norm=(float(final["parameter_gradient_norm"])
                if np.isfinite(final["parameter_gradient_norm"]) else None),
            failure=failure, seconds=perf_counter()-started)
        if blocks:
            before = arrays["refinement_block_before_fit_energy_components"][:, -1]
            after = arrays["refinement_block_after_fit_energy_components"][:, -1]
            after_r = arrays["refinement_block_after_rotation_energy_components"][:, -1]
            after_j = arrays["refinement_block_after_assignment_energy_components"][:, -1]
            report["maximum_observed_block_increases"] = dict(
                neural=float(np.max(after-before)), rotation=float(np.max(after_r-after)),
                assignment=float(np.max(after_j-after_r)))
        return arrays, report

    for step in range(steps):
        before = states[-1]["energy_components"].copy()
        fit_columns, fit_rotation = columns.copy(), rotation.copy()
        parameters_before = torch.cat([parameter.detach().reshape(-1)
                                      for parameter in model.parameters()]).clone()
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=20,
            max_eval=25, tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            z = prediction()
            values = objective_components(z, fit_columns, fit_rotation, p)
            values_np = values.detach().numpy().copy()
            finite_loss = bool(torch.isfinite(values).all())
            gradient_norm = np.nan
            finite_gradient = False
            if finite_loss:
                values[-1].backward()
                flat_gradient = torch.cat([parameter.grad.reshape(-1)
                    for parameter in model.parameters() if parameter.grad is not None])
                finite_gradient = bool(torch.isfinite(flat_gradient).all())
                gradient_norm = float(torch.linalg.vector_norm(flat_gradient).detach())
            trials.append(dict(step=step,
                mapped_points_world_m=p["target_origin"]+p["source_rms"]*z.detach().numpy(),
                target_indices=fit_columns.copy(), rotation_relation=fit_rotation.copy(),
                energy_components=values_np, parameter_gradient_norm=gradient_norm,
                finite_objective=finite_loss, finite_parameter_gradient=finite_gradient))
            if not finite_loss or not finite_gradient:
                kind = "nonfinite_objective" if not finite_loss else "nonfinite_parameter_gradient"
                failure = dict(kind=kind, step=step, trial_index=len(trials)-1,
                    nonfinite_energy_terms=[name for name, value in zip(ENERGY_TERMS, values_np)
                                           if not np.isfinite(value)])
                arrays, report = finish("failed_neural_refinement", failure)
                raise NeuralRefinementFailure(kind, arrays, report)
            return values[-1]

        returned_initial = optimizer.step(closure)
        with torch.no_grad():
            z = prediction().numpy()
            after_fit = objective_components(torch.as_tensor(z), fit_columns,
                                             fit_rotation, p).numpy()
        rotation, evidence = relation_rotation(z, p)
        after_rotation = objective_components(torch.as_tensor(z), fit_columns, rotation, p).detach().numpy()
        columns = assign_full_target(z, p)
        final = state(z, columns, rotation, evidence)
        optimizer_state = optimizer.state[next(iter(model.parameters()))]
        # The installed LBFGS returns before creating a direction/history
        # when its initial gradient satisfies tolerance_grad. That is a
        # solver outcome with no search, represented explicitly below.
        search_started = optimizer_state["n_iter"] > 0
        parameter_step = torch.cat([parameter.detach().reshape(-1)
                                    for parameter in model.parameters()])-parameters_before
        blocks.append(dict(before_fit_energy_components=before,
            after_fit_energy_components=after_fit,
            after_rotation_energy_components=after_rotation,
            after_assignment_energy_components=final["energy_components"],
            fit_target_indices=fit_columns, fit_rotation_relation=fit_rotation,
            optimizer_returned_initial_loss=float(returned_initial.detach()),
            closure_evaluations=optimizer_state["func_evals"],
            optimizer_iterations=optimizer_state["n_iter"],
            optimizer_search_started=search_started,
            optimizer_history_pairs=(len(optimizer_state["old_dirs"]) if search_started else 0),
            optimizer_last_search_step_size=(float(optimizer_state["t"]) if search_started else np.nan),
            optimizer_last_directional_derivative=(float(torch.dot(optimizer_state["d"],
                optimizer_state["prev_flat_grad"])) if search_started else np.nan),
            parameter_step_l2_norm=float(torch.linalg.vector_norm(parameter_step)),
            parameter_step_inf_norm=float(torch.max(torch.abs(parameter_step)))))
        states.append(final)
    return finish("computed_neural_refinement")
# END one full-evidence objective and its recorded alternating neural fit.
