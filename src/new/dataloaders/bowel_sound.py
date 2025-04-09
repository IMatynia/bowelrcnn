from pydantic import BaseModel, model_validator
import numpy as np


class BowelSoundRaw(BaseModel):
    """Location of a bowelsound within a file. Position depicted in seconds"""

    start: float
    end: float | None = None

    min_frequency: float | None = np.nan
    max_frequency: float | None = np.nan

    category: str | float | None = np.nan

    @model_validator(mode="before")
    @classmethod
    def check_valid_bowel_sound(cls, data: dict):
        if isinstance(data, dict):
            assert data.get("end") == None or data["end"] - data["start"] >= 0, "Bowel sounds length should be greater or equal zero"
        return data

    @property
    def midpoint(self):
        return (self.start + self.end) / 2

    @property
    def length(self):
        return self.end - self.start

    @property
    def start_end_tuple(self):
        return self.start, self.end

    def offset_by(self, offset: float):
        return BowelSoundRaw(
            start=self.start + offset,
            end=self.end + offset,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            category=self.category
        )

    def to_csv(self):
        # start, end, fmin, fmax, category
        return [self.start, self.end, self.min_frequency, self.max_frequency, self.category]

    def to_limits(self, start_flag, end_flag):
        return (self.start, start_flag, self), (self.end, end_flag, self)

    def __lt__(self, other):
        return self.midpoint < other.midpoint

    def __hash__(self):
        return hash((self.start, self.end))
