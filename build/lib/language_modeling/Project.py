from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    """
    This class represents our project.
    It stores useful information about the structure
    """

    base_dir: Path = Path(__file__).parents[2]
    data_dir = base_dir / "data"

    config_dir = base_dir / "config"
    main_config_file = config_dir / "config.yml"
    hparams_file = config_dir / "hparams.yml"
    models_dir = base_dir / 'models'

    log_dir = base_dir / "log"

    wandb_dir = log_dir / "wandb"

    export_dir = base_dir / "exports"

    def __post_init__(self) -> None:
        # create the directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        self.wandb_dir.mkdir(exist_ok=True)
        self.export_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    project = Project()
    print(project.hparams_file) 
