"""Policy baselines for the benchmark.

A *policy* is a first-class bench building block (parallel to ``robot/``,
``task/``, ``render/``): a closed-loop controller that maps observations to
robot actions. Each baseline lives in ``policy/<name>.py`` (single-file) or
``policy/<name>/`` (a subpackage, when it ships its own train launcher / server
/ docs, e.g. ``gr00t/``).

``_base.py`` defines the ``PolicyBase`` contract plus ``load_policy`` /
``add_policy_arguments`` dispatch. The (future R5) eval loop *consumes* this
package by name via ``--policy_type``; it is not nested under it.
"""
