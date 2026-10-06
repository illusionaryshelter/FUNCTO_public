"""CPU checks of explicit per-owner CLI inputs before any native allocation."""

# BEGIN repeated owner arguments must never lose an identity through dict overwrite.
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'planning'))
from run_recorded_skill import owner_voxel_sizes


def test_distinct_object_resolutions_keep_explicit_identities():
    assert owner_voxel_sizes([('small', '.0015'), ('large', '.0045')], {'large', 'small'}) == {
        'small': .0015, 'large': .0045}


@pytest.mark.parametrize('arguments', [
    [('small', '.0015'), ('small', '.0045')],
    [('small', '.0015')],
    [('small', '.0015'), ('large', '.0045'), ('extra', '.01')],
    [('small', '.0015'), ('large', 'nan')],
    [('small', '.0015'), ('large', 'inf')],
    [('small', '.0015'), ('large', '0')],
    [('small', '.0015'), ('large', '-.0045')],
])
def test_incomplete_ambiguous_or_nonfinite_cli_mapping_is_rejected(arguments):
    with pytest.raises(ValueError):
        owner_voxel_sizes(arguments, {'small', 'large'})
# END no test imports an environment, starts a planner or initializes CUDA.
