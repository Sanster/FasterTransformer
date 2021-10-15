import os

DEBUG_NAME = os.environ.get("DEBUG_NAME", "")


def is_debug_onmt() -> bool:
    if DEBUG_NAME == "ONMT":
        return True
    return False


def is_debug_master() -> bool:
    if DEBUG_NAME == "MASTER":
        return True
    return False
