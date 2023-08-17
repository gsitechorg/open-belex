r"""
By Dylon Edwards
"""

from abc import ABC, abstractmethod


class Rendition(ABC):
    pass


class Renderer(ABC):

    @abstractmethod
    def AND(self, a, b):
        raise NotImplementedError

    @abstractmethod
    def OR(self, a, b):
        raise NotImplementedError

    @abstractmethod
    def XOR(self, a, b):
        raise NotImplementedError

    @abstractmethod
    def ASSIGN(self, a, b):
        raise NotImplementedError


class Renderable(ABC):

    @abstractmethod
    def render(self: "Renderable", renderer: Renderer) -> Rendition:
        raise NotImplementedError
