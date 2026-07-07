import torch


def particle_energy_loss(
    particle_velocities: torch.Tensor,
    mass: float = 1.0,
) -> torch.Tensor:
    """
    Particle kinetic-energy stability loss.

    particle_velocities shape:
        [batch, time, particles, 3]

    The loss penalizes sudden kinetic-energy changes between frames.
    """

    kinetic_energy = 0.5 * mass * torch.sum(
        particle_velocities ** 2,
        dim=-1,
    )

    energy_change = kinetic_energy[:, 1:] - kinetic_energy[:, :-1]

    return torch.mean(energy_change ** 2)


def particle_smoothness_loss(
    particle_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Smoothness loss for particle trajectories.

    particle_positions shape:
        [batch, time, particles, 3]
    """

    velocity_like = particle_positions[:, 1:] - particle_positions[:, :-1]
    acceleration_like = velocity_like[:, 1:] - velocity_like[:, :-1]

    return torch.mean(acceleration_like ** 2)


def total_lagrangian_loss(
    particle_positions: torch.Tensor,
    particle_velocities: torch.Tensor,
    lambda_energy: float = 0.01,
    lambda_smooth: float = 0.001,
) -> torch.Tensor:
    """
    Combined Lagrangian particle constraint.
    """

    energy = particle_energy_loss(particle_velocities)
    smooth = particle_smoothness_loss(particle_positions)

    return lambda_energy * energy + lambda_smooth * smooth
