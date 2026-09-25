"""Side-fold T-shirt variant without the intermediate edge-drag stage."""

from typing import Generator

from deformablebench.bench.common.types import ExecutionStep
from deformablebench.bench.task.task_side_fold_tshirt_v2 import TshirtFoldV2Task


class TshirtFoldNoDragEdgeTask(TshirtFoldV2Task):
    """Run the original fold sequence while skipping the edge pullback."""

    def _stage_close_side_fold(self, *args, **kwargs):
        """Finish after the vertical retract, before edge-drag/home planning."""
        base_stage = super()._stage_close_side_fold(*args, **kwargs)
        try:
            while True:
                try:
                    step = next(base_stage)
                except StopIteration as stop:
                    return bool(stop.value)

                yield step
                if step.description == "side_fold_retract":
                    return True
        finally:
            base_stage.close()

    def _drag_fold_edge_inward(
        self,
        drag_dist: float = 0.15,
    ) -> Generator[ExecutionStep, None, bool]:
        # Defensive no-op in case another inherited stage requests edge-drag:
        # perform no grasp, pull, release, or stage-specific physics changes.
        yield from ()
        return True
