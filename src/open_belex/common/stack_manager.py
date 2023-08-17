r"""
By Dylon Edwards
"""

import threading
from collections import deque
from typing import Any, ClassVar, Deque, Optional, Type


class StackManager:
    THREAD_LOCAL: ClassVar = threading.local()

    @classmethod
    def named_stack(cls: Type["StackManager"], stack_name: str) -> Deque[Any]:
        thread_locals = cls.THREAD_LOCAL

        if not hasattr(thread_locals, "named_stacks"):
            thread_locals.named_stacks = {}

        named_stacks = thread_locals.named_stacks
        if stack_name not in named_stacks:
            named_stacks[stack_name] = deque()

        named_stack = named_stacks[stack_name]
        return named_stack

    @classmethod
    def push(cls: Type["StackManager"], stack_name: str, element: Any) -> Any:
        named_stack = cls.named_stack(stack_name)
        named_stack.append(element)
        return element

    @classmethod
    def pop(cls: Type["StackManager"], stack_name: str) -> Any:
        cls.assert_has_elem(stack_name)
        named_stack = cls.named_stack(stack_name)
        return named_stack.pop()

    @classmethod
    def peek(cls: Type["StackManager"], stack_name: str) -> Any:
        cls.assert_has_elem(stack_name)
        named_stack = cls.named_stack(stack_name)
        return named_stack[-1]

    @classmethod
    def size(cls: Type["StackManager"], stack_name: str) -> int:
        named_stack = cls.named_stack(stack_name)
        return len(named_stack)

    @classmethod
    def has_elem(cls: Type["StackManager"], stack_name: str) -> bool:
        return cls.size(stack_name) > 0

    @classmethod
    def assert_has_elem(cls: Type["StackManager"], stack_name: str) -> None:
        if not cls.has_elem(stack_name):
            raise AssertionError(f"Stack named [{stack_name}] is empty!")


def contextual(cls: Optional[Type] = None,
               lazy_init: bool = False) -> Type:

    def decorator(cls: Type) -> Type:
        stack_nym = f"__CONTEXT__{cls.__name__}"

        def push_context(*args, **kwargs) -> cls:
            context = cls(*args, **kwargs)
            StackManager.push(stack_nym, context)
            return context

        def has_context() -> bool:
            return StackManager.has_elem(stack_nym)

        def context() -> cls:
            if StackManager.has_elem(stack_nym):
                context = StackManager.peek(stack_nym)
            elif lazy_init:
                context = push_context()
            else:
                raise KeyError(f"No context defined for {cls.__name__}")
            return context

        def pop_context() -> cls:
            context = StackManager.pop(stack_nym)
            return context

        if not hasattr(cls, "push_context"):
            cls.push_context = push_context
        if not hasattr(cls, "has_context"):
            cls.has_context = has_context
        if not hasattr(cls, "context"):
            cls.context = context
        if not hasattr(cls, "pop_context"):
            cls.pop_context = pop_context
        cls.__STACK_NYM__ = stack_nym
        return cls

    if cls is not None:
        return decorator(cls)

    return decorator
