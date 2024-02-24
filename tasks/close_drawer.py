import logging
from .open_drawer import OpenDrawer


class CloseDrawer(OpenDrawer):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)
        self.task = 'close_drawer'
        self.grip_open = cfg.gripper_open[self.task]
        self.logger = logging.getLogger(__name__)
