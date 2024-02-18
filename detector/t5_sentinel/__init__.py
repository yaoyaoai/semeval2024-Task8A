import yaml
from detector.t5_sentinel.t5types import Config


with open('detector/t5_sentinel/settingsmultilanguagewrong.yaml', 'r') as f:
    config = Config(**yaml.safe_load(f))
