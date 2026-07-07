from pathlib import Path
import numpy as np


def export_openfoam_velocity_to_numpy(
    case_folder: str,
    output_file: str,
    grid_shape=(42, 42, 42),
):
    """
    Placeholder OpenFOAM export pipeline.

    Full version should:
    1. read OpenFOAM velocity field U
    2. interpolate U onto a regular 3D grid
    3. save the result as [X, Y, Z, 3]
    """

    case_folder = Path(case_folder)
    output_file = Path(output_file)

    print(f"OpenFOAM case folder: {case_folder}")

    velocity_field = np.zeros(
        (*grid_shape, 3),
        dtype=np.float32,
    )

    np.save(output_file, velocity_field)

    print(f"Saved velocity field to: {output_file}")
    print(f"Velocity field shape: {velocity_field.shape}")


if __name__ == "__main__":
    export_openfoam_velocity_to_numpy(
        case_folder="openfoam_case",
        output_file="velocity_field.npy",
        grid_shape=(42, 42, 42),
    )
