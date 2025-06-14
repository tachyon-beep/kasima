import sys
import types

import importlib
import os


def test_clearml_initialisation(monkeypatch):
    # Create a stub clearml module with a Task class
    clearml_stub = types.ModuleType("clearml")

    class DummyTask:
        init_args = None

        @classmethod
        def init(cls, project_name, task_name):
            cls.init_args = (project_name, task_name)
            return cls

        @classmethod
        def current_task(cls):
            return True

    clearml_stub.Task = DummyTask
    monkeypatch.setitem(sys.modules, "clearml", clearml_stub)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    if "scripts.run_experiment" in sys.modules:
        del sys.modules["scripts.run_experiment"]
    import scripts.run_experiment as _  # noqa: F401

    assert DummyTask.init_args == ("kasima-cifar", "two-spirals")
    assert DummyTask.current_task()

