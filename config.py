from pathlib import Path


def load_config(config_path="config.txt"):
    values = {}

    with open(config_path, "r", encoding="utf-8") as config_file:
        for line_number, line in enumerate(config_file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(
                    f"Invalid config entry on line {line_number}: {line}"
                )

            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    required = ["image_dir", "n_workers"]

    for key in required:
        if key not in values or not values[key]:
            raise ValueError(f"Missing required config value: {key}")

    image_dir = Path(values["image_dir"]).expanduser()

    try:
        n_workers = int(values["n_workers"])
    except ValueError:
        raise ValueError("n_workers must be a positive integer.")

    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {image_dir}")

    return {
        "image_dir": image_dir,
        "n_workers": n_workers,
    }
