# Project structure

The project is divided into several folders, below is the description of every folder

- `config`: Holds the configuration files that are needed the project to run. The contents of every file is very self explanatory.
  - `config.yml`: Configuration of the logger of the project.
  - `hparams.yml`: Hyper parameters file used for training the model.
  - `api.yml`: Parameters used for the API.

- `data`: Any input data that is used by the model. This folder will host the training data to train the language model.
- `exports`: Holds all the files/folders exported by the model. Model checkpoints during training will be put here.
- `logs`: Holds all the log files from the system.
- `notebooks`: Holds all the jupyter notebooks used.
- `scripts`: Helper bash scripts to call the python code.
- `tests`: Holds all the unit tests for the project
- `src`: All the source code for this project
  - `src/training`: Holds all the code needed to train the language model
  - `src/apis`: Holds all the code needed to run the bert-as-a-service API.
  - 