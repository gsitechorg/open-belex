r"""
 By Brian Beckman and Dylon Edwards

 Copyright 2023 GSI Technology, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the “Software”), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from typing import List, Optional, Set


class StringArena:
    r"""A pool (arena) of locally unique string resources.

    TODO: context manager for arenas!
    TODO: inheriting from Dict did not work; don't know why not.
    """

    def __init__(self: "StringArena",
                 prefix: str,
                 nregs: int,
                 reservations: Optional[Set[int]]) -> None:
        if reservations is None:
            reservations = set()
        regs = [f"{prefix}{i}" for i in range(nregs)
                if i not in reservations]
        self.arena = dict(zip(regs, nregs * [False]))

    @property
    def nregs(self: "StringArena") -> int:
        return len(self.arena)

    def allocated_count(self) -> int:
        temp = [v for v in self.arena.values() if v]
        result = len(temp)
        return result

    def free_count(self) -> int:
        temp = [v for v in self.arena.values() if not v]
        result = len(temp)
        return result

    def allocate_several(self, n: int) -> List[str]:
        """Allocate a bunch; if any failure, free any and return None."""
        result = [self.allocate() for _ in range(n)]
        if not all(result):
            for reg in result:
                self.free(reg)
            result = None
        return result

    def free_several(self, several: List[str]) -> None:
        for s in several:
            self.free(s)
        pass

    def free_all(self) -> None:
        self.free_several(list(self.arena.keys()))

    def allocate(self) -> str:
        """Return the first free string resource found and 'None' if
        there are no free string resources left."""
        result = None
        for k, v in self.arena.items():
            if not v:
                self.arena[k] = True
                result = k
                break
        return result

    def reserve(self, k) -> None:
        if k is None or k not in self.arena:
            return

        if self.arena[k]:
            raise ValueError(f"{k} is already reserved")

        if k not in self.arena:
            raise KeyError(f"Unknown key: {k}")

        self.arena[k] = True

    def free(self, k) -> None:
        """Free up a string resource no longer needed. Ignore 'None'."""
        if k is None:
            return

        if not self.arena[k]:
            # Ignore double frees (because duplicates legitimately
            # exist for SM_REGs and their inverses.
            pass

        self.arena[k] = False
