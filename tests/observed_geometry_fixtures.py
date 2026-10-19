"""Small explicit ordinary distance fields for CPU model tests."""
# BEGIN the fixture no longer constructs source solids or native component trees.
import numpy as np
from manipulation.online_geometry import FrozenESDF


def field_from_samples(values, *, origin, voxel_size, observed=None, owner_ids=(), source='test source'):
    """Freeze declared distance samples; observation masks belong to the scene."""
    return FrozenESDF(np.asarray(values, np.float32), origin=origin, voxel_size=voxel_size,
                      owner_ids=owner_ids, source=source)
# END no old source-geometry fixture remains as a compatibility adapter.
