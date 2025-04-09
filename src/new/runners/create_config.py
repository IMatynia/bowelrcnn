from src.new.config.examples_code.config_best import get_big_spec_baseline
from src.new.config.constants import PROJECT_ROOT

if __name__ == "__main__":
    destination = PROJECT_ROOT / "configs" / "example_high_res.json"

    # Combination
    config = get_big_spec_baseline()

    destination.write_text(config.model_dump_json(indent=1))
