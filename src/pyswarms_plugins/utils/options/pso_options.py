from dataclasses import dataclass
import dataclasses
from typing import Dict, Optional, Union

__all__ = ["PSOOptions", "GeneralOptimizerPSOOptions"]
__author__ = "Hung-Tien Huang"
__contact__ = "hungtienhuang@gmail.com"


@dataclass
class PSOOptions:
    """     
    Attributes:
        w (float): inertia parameter. if used with the :code:`Ring`, :code:`VonNeumann` or :code:`Random` topology the additional parameter k must be included
        c1 (float): cognitive parameter
        c2 (float): social parameter
    """
    w: float = dataclasses.field(init=True)
    c1: float = dataclasses.field(init=True)
    c2: float = dataclasses.field(init=True)

    def create_options_dict(self) -> Dict[str, float]:
        """Create options dictionary for upstream library.

        Returns:
            Dict[str, float]: options dictionary.
        """
        options_dict: Dict[str, float] = {
            "w": self.w,
            "c1": self.c1,
            "c2": self.c2
        }
        return options_dict


@dataclass
class GeneralOptimizerPSOOptions(PSOOptions):
    """
    Attributes:
        c1 (float): cognitive parameter
        c2 (float): social parameter
        w (float): inertia parameter. if used with the :code:`Ring`, :code:`VonNeumann` or :code:`Random` topology the additional parameter k must be included
        k (int): number of neighbors to be considered. Must be a positive integer less than :code:`n_particles` if used with the :code:`Ring` topology the additional parameters k and p must be included.
        p (int): in range {1,2}. the Minkowski p-norm to use. 1 is the sum-of-absolute values (or L1 distance) while 2 is the Euclidean (or L2) distance. if used with the :code:`VonNeumann` topology the additional parameters p and r must be included
        r (int): the range of the VonNeumann topology.  This is used to determine the number of neighbours in the topology.
    """

    k: Optional[int] = dataclasses.field(default=None, init=True)
    p: Optional[int] = dataclasses.field(default=None, init=True)
    r: Optional[int] = dataclasses.field(default=None, init=True)

    def create_options_dict(self) -> Dict[str, Union[float, int]]:
        """Create options dictionary for upstream library.

        Returns:
            Dict[str, float]: options dictionary.
        """
        options_dict: Dict[str, Union[float,
                                      int]] = super().create_options_dict()
        if self.k is not None:
            options_dict["k"] = self.k
        if self.p is not None:
            options_dict["p"] = self.p
        if self.r is not None:
            options_dict["r"] = self.r
        return options_dict
