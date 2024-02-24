import logging
from .open_cabinet import OpenCabinet


class CloseCabinet(OpenCabinet):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)
        self.task = 'close_cabinet'
        self.grip_open = cfg.gripper_open[self.task]
        self.logger = logging.getLogger(__name__)
