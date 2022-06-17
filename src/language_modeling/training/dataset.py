# Project dependencies
from transformers.data.datasets import LineByLineTextDataset
from datasets import load_dataset

# Project imports
from language_modeling.Project import Project
from language_modeling.util import get_default_logger, load_hparams

hparams = load_hparams()

logger = get_default_logger()


def get_dataset():
    logger.info("Loading dataset")
    data_file = Project.data_dir / hparams.dataset_file_name
    dataset = load_dataset("text", data_files={"train": str(data_file)})
    logger.info("Done Loading dataset")
    return dataset


if __name__ == "__main__":
    get_dataset()
