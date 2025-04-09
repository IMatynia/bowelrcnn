from pathlib import Path

from pydantic import ValidationError
from src.new.dataloaders.bowel_sound import BowelSoundRaw
import pandas as pd
import logging as lg
logging = lg.getLogger("csv-handler")


class BowelSoundCSVFileHandler:
    def __init__(self, filename: Path):
        self.filename = filename
        self.cache = None

    def cached(self):
        if self.cache is None:
            self.cache = list(self.load())
        return self.cache

    def load(self):
        all_bowel_sounds_df = pd.read_csv(self.filename)
        for bs in all_bowel_sounds_df.values:
            bs = list(bs) + [None] * 10
            try:
                yield BowelSoundRaw(
                    start=bs[0],
                    end=bs[1],
                    min_frequency=bs[2],
                    max_frequency=bs[3],
                    category=bs[4]
                )
            except ValidationError as e:
                logging.error(f"Invalid bowel sound: {e}")

    def save(self, bowel_sounds: list[BowelSoundRaw]):
        df = pd.DataFrame(
            data=[bs.to_csv() for bs in bowel_sounds],
            columns=["start", "end", "fmin", "fmax", "category"]
        )
        df.to_csv(self.filename, index=False)
