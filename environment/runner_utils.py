# https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
# pip install nvidia-ml-py3
# import nvidia_smi
import carb
import json
import typing
import logging
from omni.isaac.kit import SimulationApp


def get_available_vrams():
    # https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    # pip install nvidia-ml-py3
    return 20
    
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total/1e9)
    print("Free memory:", info.free/1e9)
    print("Used memory:", info.used/1e9)
    free_memeory = info.free/(1e9)
    nvidia_smi.nvmlShutdown()
    return free_memeory


def get_simulation(simulation_app=None, simulation_context=None, headless=True, gpu_id=0, dt=1.0/120.0, high_quality=False):
    _headless = headless
    new_simulation = False

    if simulation_app is None:
        if high_quality:
            # for high quality path tracing
            simulation_app = SimulationApp({"headless": _headless, "anti_aliasing": 0, 'active_gpu':gpu_id, 'multi_gpu': False,
                                            "renderer": "PathTracing", "samples_per_pixel_per_frame": 32, "max_bounces": 32})
        else:
            simulation_app = SimulationApp({"headless": _headless, "anti_aliasing": 0, 'active_gpu':gpu_id, 'multi_gpu': False})

        logging.getLogger("omni.hydra").setLevel(logging.ERROR)
        logging.getLogger("omni.isaac.urdf").setLevel(logging.ERROR)
        logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

        l = carb.logging.LEVEL_ERROR
        carb.settings.get_settings().set("/log/level", l)
        carb.settings.get_settings().set("/log/fileLogLevel", l)
        carb.settings.get_settings().set("/log/outputStreamLevel", l)

        new_simulation = True

        from omni.isaac.core import SimulationContext
        simulation_context = SimulationContext(stage_units_in_meters=0.01)
        simulation_context.initialize_physics()
        simulation_context.set_simulation_dt(physics_dt=dt, rendering_dt=dt)
        return simulation_app, simulation_context, new_simulation

    else:
        free_memeory = get_available_vrams()
        if free_memeory < 1:
            #well, this workaround with memory leak is not working.....
            # it seems when nvdia shut down isaac , it does not unbind certain resources
            print("not enough memory need new simulation")
            
            simulation_app.close()
            exit()
            
            simulation_app = SimulationApp({"headless": _headless, "anti_aliasing": 1})
            
            new_simulation = True

            from omni.isaac.core import SimulationContext
            simulation_context = SimulationContext(stage_units_in_meters=0.01)
            simulation_context.initialize_physics()
            simulation_context.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0)
            return simulation_app, simulation_context, new_simulation
        
        else:
            return simulation_app, simulation_context, new_simulation
