from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

__all__ = ["AccelerationStrategy", "BasicAccelerationStrategy"]


class AccelerationStrategy(ABC):

    @abstractmethod
    def exploration_strategy(self, c1: float, c2: float) -> Tuple[float, float]:
        pass

    @abstractmethod
    def exploitation_strategy(self, c1: float,
                              c2: float) -> Tuple[float, float]:
        pass

    @abstractmethod
    def convergence_strategy(self, c1: float, c2: float) -> Tuple[float, float]:
        pass

    @abstractmethod
    def jumping_out_strategy(self, c1: float, c2: float) -> Tuple[float, float]:
        pass


class BasicAccelerationStrategy(AccelerationStrategy):

    __c_sum: float
    __c_clamp: Tuple[float, float]
    __acc_rate: Optional[float]
    __acc_clamp: Tuple[float, float]
    __force_normalize: bool

    def __init__(self,
                 c_sum: float = 4.0,
                 c_clamp: Tuple[float, float] = (1.5, 2.5),
                 acc_rate: Optional[float] = None,
                 acc_clamp: Tuple[float, float] = (0.05, 0.1),
                 force_normalize: bool = False) -> None:
        super().__init__()
        self.__c_sum = c_sum
        self.__c_clamp = c_clamp
        self.__acc_rate = acc_rate
        self.__acc_clamp = acc_clamp
        self.__force_normalize = force_normalize

    def exploration_strategy(self, c1: float, c2: float) -> Tuple[float, float]:
        acc_rate: float
        if self.__acc_rate is not None:
            acc_rate = self.__acc_rate
        else:
            acc_rate = self.__generate_acc_rate()
        c1 = c1 + acc_rate
        c2 = c2 - acc_rate
        c1 = self.__clamp_c(c1)
        c2 = self.__clamp_c(c2)
        if self.__force_normalize or c1 + c2 > self.__c_sum:
            return self.__normalize(c1, c2)
        return c1, c2

    def exploitation_strategy(self, c1: float,
                              c2: float) -> Tuple[float, float]:
        acc_rate: float
        if self.__acc_rate is not None:
            acc_rate = self.__acc_rate
        else:
            acc_rate = self.__generate_acc_rate()
        acc_rate = 0.5 * acc_rate
        c1 = c1 + acc_rate
        c2 = c2 - acc_rate
        c1 = self.__clamp_c(c1)
        c2 = self.__clamp_c(c2)
        if self.__force_normalize or c1 + c2 > self.__c_sum:
            return self.__normalize(c1, c2)
        return c1, c2

    def convergence_strategy(self, c1: float, c2: float) -> Tuple[float, float]:
        acc_rate: float
        if self.__acc_rate is not None:
            acc_rate = self.__acc_rate
        else:
            acc_rate = self.__generate_acc_rate()
        acc_rate = 0.5 * acc_rate
        c1 = c1 + acc_rate
        c2 = c2 + acc_rate
        c1 = self.__clamp_c(c1)
        c2 = self.__clamp_c(c2)
        if self.__force_normalize or c1 + c2 > self.__c_sum:
            return self.__normalize(c1, c2)
        return c1, c2

    def jumping_out_strategy(self, c1: float, c2: float) -> Tuple[float, float]:
        acc_rate: float
        if self.__acc_rate is not None:
            acc_rate = self.__acc_rate
        else:
            acc_rate = self.__generate_acc_rate()
        c1 = c1 - acc_rate
        c2 = c2 + acc_rate
        c1 = self.__clamp_c(c1)
        c2 = self.__clamp_c(c2)
        if self.__force_normalize or c1 + c2 > self.__c_sum:
            return self.__normalize(c1, c2)
        return c1, c2

    def __normalize(self, c1: float, c2: float) -> Tuple[float, float]:
        c1 = c1 / (c1 + c2) * self.__c_sum
        c2 = self.__c_sum - c1
        return c1, c2

    def __generate_acc_rate(self) -> float:
        acc_rate: float = np.random.uniform(low=self.__acc_clamp[0],
                                            high=self.__acc_clamp[1])
        return acc_rate

    def __clamp_c(self, c: float) -> float:
        if c < self.__c_clamp[0]:
            c = self.__c_clamp[0]
        elif self.__c_clamp[1] < c:
            c = self.__c_clamp[1]
        return c