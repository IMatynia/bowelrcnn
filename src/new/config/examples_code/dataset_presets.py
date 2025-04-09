from src.new.config.constants import PROJECT_ROOT
from src.new.preprocessing.random_data_split import RandomDatasetSplit
from src.new.config.cnn_config import *
from src.new.config.model_config import *
from src.new.config.constants import PROJECT_ROOT


def random_dataset_7_2_1(dataset_root, audio_properties):
    split = RandomDatasetSplit(dataset_root, 0.7, 0.2, 0.1, audio_properties)
    return split.get_random_data_setup()


def make_dataset_from_old(audio_properties):
    train = [
        "annotation-shortened-1405.wav",
        "agrest-M200516-1-2-part20210122.wav",
        "borowka-M8-201102-000_201103-part20210123.wav",
        "brzoskwinia-M200505-6-1-part20210325.wav",
        "pomelo-20210310-part20210325-2.wav",
    ]
    valid = [
        "poziomka-REVISED-annotation20200711-from-3-200311.wav",
        "gruszka-M200505-2-1-fragment.wav",
        "jablko-M200516-2-2-S.wav",
        "jagoda-M200606-8-2-S.wav",
        "porzeczka-M200507-6-0-S.wav",
        "poziomka-M200311-3-0-S.wav",
        "morela-M200613-6-2-S.wav",
    ]
    test = [
        "annotation20200711-train.wav",
        "czeresnia-M1-201127-000_201128-part20210326.wav",
        "truskawka-M200505-1-1-S.wav",
        "pigwa-M6-201111-000_201122-S.wav",
        "malina-M200521-6-2-S.wav",
        "sliwka-M200606-2-2-S.wav",
        "mango-M200505-4-2-S.wav",
    ]
    return DatasetSetup(audio_properties=audio_properties,
                        train_files=[BowelSoundSampleFile(sample_id=file[:-4]) for file in train],
                        valid_files=[BowelSoundSampleFile(sample_id=file[:-4]) for file in valid],
                        test_files=[BowelSoundSampleFile(sample_id=file[:-4]) for file in test])
