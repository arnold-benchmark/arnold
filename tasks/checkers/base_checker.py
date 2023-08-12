import omni
import carb
from pxr import PhysxSchema, UsdPhysics


class BaseChecker():

    def __init__(self) -> None:
        """
        ::params:
            :run_time: is run-time task checker or not
        """
        
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
       
        self.success_steps = 0
        self.success = False
        self.time = 0
       
        # log
        self.total_step = 0
        self.print_every = 240
        self.checking_interval = 15
        self.is_init = False

        # get time per second
        physicsScenePath = "/physicsScene"
        scene = UsdPhysics.Scene.Get(self.stage, physicsScenePath)
        if not scene:
            carb.log_warn("physics scene not found")
            raise Exception("physics scene not found")

        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        self.steps_per_second = physxSceneAPI.GetTimeStepsPerSecondAttr().Get()
        
    def initialization_step(self):
        self.is_init = True
        self.create_task_callback()
    
    def get_diff(self):
        raise NotImplementedError

    def create_task_callback(self):
        stream = self.timeline.get_timeline_event_stream()
        
        self._timeline_subscription = stream.create_subscription_to_pop(self._on_timeline_event)
        # subscribe to Physics updates:
        self._physics_update_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )

    def _on_timeline_event(self, e):
        """
        set up timeline event
        """
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            self.it = 0
            self.time = 0
            self.reset()
    
    def reset(self):
        """
        Reset event
        """
        self._physics_update_subscription = None
        self._timeline_subscription = None
        # self._setup_callbacks()
    
    def _on_success_hold(self):
        pass
    
    def _on_success(self):
        carb.log_info("task sucess")
        self.success = True

    def _on_not_success(self):
        # carb.log_info("task not sucess")
        self.success_steps = 0
        self.success = False
      

    def _on_physics_step(self, dt):
        """
        Physics event
        """
        self.time += 1
        self.start_checking()
        
    
    def start_checking(self):
        if self.success_steps > self.steps_per_second * 2:
            self._on_success()
