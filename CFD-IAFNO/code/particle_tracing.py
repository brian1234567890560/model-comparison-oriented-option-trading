import numpy as np


def trilinear_interpolate_velocity(
    velocity_field: np.ndarray,
    position: np.ndarray,
):
    """
    Interpolate velocity at one particle position.

    velocity_field shape:
        [nx, ny, nz, 3]

    position:
        normalized position inside grid coordinates:
        [x, y, z]
    """

    nx, ny, nz, _ = velocity_field.shape

    x, y, z = position

    x = np.clip(x, 0, nx - 1.001)
    y = np.clip(y, 0, ny - 1.001)
    z = np.clip(z, 0, nz - 1.001)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    z0 = int(np.floor(z))

    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    z1 = min(z0 + 1, nz - 1)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c000 = velocity_field[x0, y0, z0]
    c100 = velocity_field[x1, y0, z0]
    c010 = velocity_field[x0, y1, z0]
    c110 = velocity_field[x1, y1, z0]
    c001 = velocity_field[x0, y0, z1]
    c101 = velocity_field[x1, y0, z1]
    c011 = velocity_field[x0, y1, z1]
    c111 = velocity_field[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    velocity = c0 * (1 - zd) + c1 * zd

    return velocity


def advect_particles(
    particles: np.ndarray,
    velocity_field: np.ndarray,
    dt: float,
    obstacle_mask=None,
):
    """
    Advect particles through a 3D velocity field.

    particles shape:
        [particles, 3]

    velocity_field shape:
        [nx, ny, nz, 3]

    obstacle_mask:
        optional boolean grid.
        True means obstacle/wall cell.
    """

    new_particles = particles.copy()

    for i, position in enumerate(particles):
        velocity = trilinear_interpolate_velocity(
            velocity_field,
            position,
        )

        candidate = position + velocity * dt

        candidate = np.clip(
            candidate,
            [0, 0, 0],
            np.array(velocity_field.shape[:3]) - 1.001,
        )

        if obstacle_mask is not None:
            cell = tuple(candidate.astype(int))

            if obstacle_mask[cell]:
                candidate = position

        new_particles[i] = candidate

    return new_particles
