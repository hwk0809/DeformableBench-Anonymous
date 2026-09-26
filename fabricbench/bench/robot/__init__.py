"""Robot motion planning (IK + planning + skills).

pinocchio-based IK; cuRobo backend intentionally not ported (lazy import in
PlannerManager fails only if planner_type='curobo' is selected). Any entry point
that imports this package must call ``fabricbench._env.ensure_clean_ld_path()``
first (ROS boost on LD_LIBRARY_PATH shadows pinocchio's bundled boost).
"""
