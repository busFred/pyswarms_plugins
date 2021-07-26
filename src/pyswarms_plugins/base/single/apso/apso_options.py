from dataclasses import dataclass
import dataclasses
from ....utils.options.pso_options import PSOOptions

__all__ = ["APSOOptions"]
__author__ = "Hung-Tien Huang"
__contact__ = "hungtienhuang@gmail.com"

@dataclass
class APSOOptions(PSOOptions):
    sigma_min: float = dataclasses.field(default=0.1, init=True)
    sigma_max: float = dataclasses.field(default=1.0, init=True)
