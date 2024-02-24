import logging
from .pickup_object import PickupObject


class ReorientObject(PickupObject):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)
        self.task = 'reorient_object'
        self.grip_open = cfg.gripper_open[self.task]
        self.logger = logging.getLogger(__name__)
