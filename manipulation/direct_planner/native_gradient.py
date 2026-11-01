"""Compiled nearest-point distance derivatives in the reference frame.

Inputs retain Coal's numeric record (distance, two nearest points, normal),
both reference-frame link origins, both LOCAL_WORLD_ALIGNED Jacobians after
calibration, and the structural common-ancestor mask. The output is the
signed distance derivative in the original complete named-joint order.
Numba's CPU ufunc dispatch handles scalar/batch leading axes and array strides.
"""

from numba import guvectorize


@guvectorize(
    ['void(float64[:], float64[:], float64[:], float64[:,:], float64[:,:], boolean[:], float64[:])'],
    '(m),(d),(d),(k,n),(k,n),(n)->(n)', nopython=True, cache=True, target='cpu',
    fastmath=False,
)
def native_joint_gradient(numeric, first_origin, second_origin, first_jacobian,
                          second_jacobian, common_ancestor, result):
    """Pull the same rigid covectors through the two original frame Jacobians.

    For nearest point x and frame origin o, the angular covector is
    (x-o) cross normal. The first body's contribution is negative, the second
    positive. Their common ancestors generate one common rigid motion and
    have exactly zero relative-distance derivative. Arithmetic reassociation
    and fused fast-math transformations are disabled.
    """
    nx, ny, nz = numeric[7], numeric[8], numeric[9]
    ax, ay, az = (numeric[1]-first_origin[0], numeric[2]-first_origin[1],
                  numeric[3]-first_origin[2])
    bx, by, bz = (numeric[4]-second_origin[0], numeric[5]-second_origin[1],
                  numeric[6]-second_origin[2])
    atx, aty, atz = ay*nz-az*ny, az*nx-ax*nz, ax*ny-ay*nx
    btx, bty, btz = by*nz-bz*ny, bz*nx-bx*nz, bx*ny-by*nx
    for joint in range(result.shape[0]):
        if common_ancestor[joint]:
            result[joint] = 0.
        else:
            first_linear = (nx*first_jacobian[0, joint]+ny*first_jacobian[1, joint]
                            +nz*first_jacobian[2, joint])
            first_angular = (atx*first_jacobian[3, joint]+aty*first_jacobian[4, joint]
                             +atz*first_jacobian[5, joint])
            second_linear = (nx*second_jacobian[0, joint]+ny*second_jacobian[1, joint]
                             +nz*second_jacobian[2, joint])
            second_angular = (btx*second_jacobian[3, joint]+bty*second_jacobian[4, joint]
                              +btz*second_jacobian[5, joint])
            result[joint] = -(first_linear+first_angular)+(second_linear+second_angular)
