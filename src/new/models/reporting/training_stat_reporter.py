import wandb
import wandb.wandb_run
from pathlib import Path


class TrainingStatReporterBase:
    def __init__(self, name: str, config: dict, wandb_enabled: bool = False) -> None:
        self._wandb_enabled = wandb_enabled
        self._wandb_run: wandb.wandb_run.Run | None = None
        self._config = config
        self.name = name

        if wandb_enabled:
            self.init_wandb(config)

    def init_wandb(self):
        wandb.login()

    def log(self, data: dict):
        if self._wandb_enabled:
            self._wandb_run.log(data)

    def save(self, file: Path):
        if self._wandb_enabled:
            artifact = wandb.Artifact("model_weights", type="model")
            artifact.add_file(file)
            self._wandb_run.log_artifact(artifact)

    def finish(self):
        if self._wandb_enabled:
            self._wandb_run.finish()


class PatternModelTrainingReporter(TrainingStatReporterBase):
    def init_wandb(self, config: dict):
        if self._wandb_enabled == False:
            return
        super().init_wandb()
        self._wandb_run = wandb.init(
            project="InzBowelModelRCNN - Pattern model",
            name=self.name,
            notes="Model for detecting the bounding box of the bowel sound",
            config=config
        )


class ClassificationModelTrainingReporter(TrainingStatReporterBase):
    def init_wandb(self, config: dict):
        if self._wandb_enabled == False:
            return
        super().init_wandb()
        self._wandb_run = wandb.init(
            name=self.name,
            project="InzBowelModelRCNN - Classification model",
            notes="Model for detecting the bounding box of the bowel sound",
            config=config
        )
