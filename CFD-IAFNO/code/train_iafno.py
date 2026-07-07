import torch
import torch.nn.functional as F

from iafno_model import SimpleIAFNO
from particle_lagrangian_loss import total_lagrangian_loss


def train_step(
    model,
    batch,
    optimizer,
    lambda_energy: float = 0.01,
    lambda_smooth: float = 0.001,
):
    """
    One training step.

    batch dictionary:
        input_field:
            [B, X, Y, Z, 3]

        target_field:
            [B, X, Y, Z, 3]

        particle_positions:
            [B, T, P, 3]

        particle_velocities:
            [B, T, P, 3]
    """

    input_field = batch["input_field"]
    target_field = batch["target_field"]

    particle_positions = batch["particle_positions"]
    particle_velocities = batch["particle_velocities"]

    predicted_field = model(input_field)

    field_loss = F.mse_loss(predicted_field, target_field)

    lagrangian_loss = total_lagrangian_loss(
        particle_positions=particle_positions,
        particle_velocities=particle_velocities,
        lambda_energy=lambda_energy,
        lambda_smooth=lambda_smooth,
    )

    total_loss = field_loss + lagrangian_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": float(total_loss.detach().cpu()),
        "field_loss": float(field_loss.detach().cpu()),
        "lagrangian_loss": float(lagrangian_loss.detach().cpu()),
    }


def main():
    model = SimpleIAFNO(
        in_channels=3,
        hidden_channels=32,
        depth=4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    print("IAFNO model created.")
    print(model)


if __name__ == "__main__":
    main()
