"""
Mock nvtx module for when pynvtx is not available.
Provides no-op implementations of nvtx functions.
"""


def push_range(message=None, color=None, category=None):
    """No-op push_range"""
    pass


def pop_range():
    """No-op pop_range"""
    pass


class _RangeContext:
    """No-op range context manager"""
    def __init__(self, message=None, color=None, category=None):
        self.message = message
        self.color = color
        self.category = category

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def range_push(message=None, color=None, category=None):
    """Alias for push_range"""
    pass


def range_pop():
    """Alias for pop_range"""
    pass
