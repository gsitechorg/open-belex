r"""
By Dylon Edwards
"""

import click

from open_belex.bleir.generators import (IndexMarkdownGenerator,
                                         ListenerMarkdownGenerator,
                                         TransformerMarkdownGenerator,
                                         VisitorMarkdownGenerator)
from open_belex.bleir.types import Snippet

GENERATORS = [
    IndexMarkdownGenerator(),
    VisitorMarkdownGenerator(),
    ListenerMarkdownGenerator(),
    TransformerMarkdownGenerator(),
]


@click.command()
def main():
    """Generates documentation for the project."""

    global GENERATORS
    for generator in GENERATORS:
        generator.visit_bleir(Snippet)


if __name__ == "__main__":
    main()
