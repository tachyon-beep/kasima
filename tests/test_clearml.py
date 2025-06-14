import importlib
import os
import sys

import clearml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_task_init_called(monkeypatch):
    called = {}

    def fake_init(project_name, task_name):
        called["project"] = project_name
        called["task"] = task_name
        clearml.Task.current_task = staticmethod(lambda: "dummy")
        return "dummy"

    monkeypatch.setattr(clearml.Task, "init", fake_init)
    if "scripts.run_experiment" in sys.modules:
        del sys.modules["scripts.run_experiment"]
    module = importlib.import_module("scripts.run_experiment")
    assert called["project"] == "kasima-cifar"
    assert called["task"] == "run_experiment"
    assert module.Task.current_task() == "dummy"
