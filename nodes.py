import numpy as np
from _operator import xor
from threading import Event, Thread
from py_search.informed import best_first_search
from comfy import utils
from .wcf import WFC_Sample, WFC_Problem, ShareableList, ndarray


def waiting_loop(abort_loop_event: Event, pbar: utils.ProgressBar, total_steps, shm_name, ntasks=1):
    """
        Listens for interrupts and propagates to Problem running using interruption_proxy.
    Updates progress_bar via ticker_proxy, updated within Problem instances.

    @param abort_loop_event: to be triggered in the main thread once the problem(s) have been solved
    @param pbar: comfyui progress bar to update every 100 milliseconds
    @param total_steps: the total number of nodes to process in the problem(s)
    @param shm_name: shared memory name for a shared list with the elements: [ Boolean, Integer, Integer... ], in this respective order
    """
    from time import sleep
    from comfy.model_management import processing_interrupted
    shm_list = ShareableList(name=shm_name)
    while not abort_loop_event.is_set():
        sleep(.1)  # pause for 1 second
        if processing_interrupted():
            shm_list[0] = True
            return
        pbar.update_absolute(sum(list(shm_list)[1:ntasks+1]), total_steps)
    return


def terminate_generation(finished_event, shm_list, pbt: Thread):
    finished_event.set()
    pbt.join()
    interrupted = shm_list[0]
    shm_list.shm.close()
    shm_list.shm.unlink()
    if interrupted:
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()


class WFC_SampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "img_batch": ("IMAGE",),
                    "tile_width": ("INT", {"default": 32, "min": 1, "max": 128}),
                    "tile_height": ("INT", {"default": 32, "min": 1, "max": 128}),
                    "output_tiles": ("BOOLEAN", {"default": False})
                },
        }

    RETURN_TYPES = ("WFC_Sample", "IMAGE",)
    RETURN_NAMES = ("sample", "unique_tiles",)
    FUNCTION = "compute"
    CATEGORY = "Bmad/WFC"

    def compute(self, img_batch, tile_width, tile_height, output_tiles):
        import torch

        samples = [np.clip(255. * img_batch[i].cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                   for i in range(img_batch.shape[0])]
        sample = WFC_Sample(samples, tile_width, tile_height)

        if output_tiles:
            tiles = [torch.from_numpy(tile.astype(np.float32) / 255.0).unsqueeze(0)
                     for tile, freq in sample.get_tile_data().values()]
            tiles = torch.concat(tiles)
        else:
            tiles = torch.empty((1, 1, 1))

        return (sample, tiles,)


class WFC_GenerateNode:
    @staticmethod
    def NODE_INPUT_TYPES():
        return {
            "required":
                {
                    "sample": ("WFC_Sample",),
                    "starting_state": ("WFC_State",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "max_freq_adjust": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": .01}),
                    "use_8_cardinals": ("BOOLEAN", {"default": False}),
                    "relax_validation": ("BOOLEAN", {"default": False}),
                    "plateau_check_interval": ("INT", {"default": -1, "min": -1, "max": 10000}),
                },
            "optional":
                {
                    "custom_temperature_config": ("WFC_TemperatureConfig",),
                    "custom_node_value_config": ("WFC_NodeValueConfig",)
                }
        }

    @classmethod
    def INPUT_TYPES(cls):
        return cls.NODE_INPUT_TYPES()

    RETURN_TYPES = ("WFC_State",)
    RETURN_NAMES = ("state", "unique_tiles",)
    FUNCTION = "compute"
    CATEGORY = "Bmad/WFC"

    def compute(self, custom_temperature_config=None, custom_node_value_config=None, **kwargs):
        if custom_temperature_config is not None:
            kwargs.update(custom_temperature_config)

        if custom_node_value_config is not None:
            kwargs.update(custom_node_value_config)

        # prepare stuff to process interrupts & update bar
        # TODO count is also done inside Problem, maybe should use as optional arg to avoid repeating the operation
        ss = kwargs["starting_state"]
        total_tiles_to_proc = ss.size - np.count_nonzero(ss)
        if total_tiles_to_proc == 0:
            return (ss,)

        shm_list = ShareableList([False, int(0)])
        shm_name = shm_list.shm.name
        finished_event = Event()
        pbar: utils.ProgressBar = utils.ProgressBar(total_tiles_to_proc)

        t = Thread(target=waiting_loop, args=(finished_event, pbar, total_tiles_to_proc, shm_name))
        t.start()

        result = generate_single(shm_name, kwargs)

        terminate_generation(finished_event, shm_list, t)
        return (result,)


class WFC_Encode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "img": ("IMAGE",),
                    "sample": ("WFC_Sample",),
                },
        }

    RETURN_TYPES = ("WFC_State",)
    RETURN_NAMES = ("state",)
    FUNCTION = "compute"
    CATEGORY = "Bmad/WFC"

    def compute(self, img, sample: WFC_Sample):
        samples = [np.clip(255. * img[i].cpu().numpy().squeeze(), 0, 255).astype(np.uint8) for i in range(img.shape[0])]
        encoded = sample.img_to_tile_encoded_world(samples[0])  # no batch enconding, only a single image is encoded
        return (encoded,)


class WFC_Decode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "state": ("WFC_State",),
                    "sample": ("WFC_Sample",),
                },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "compute"
    CATEGORY = "Bmad/WFC"

    def compute(self, state, sample: WFC_Sample):
        import torch

        img, mask = sample.tile_encoded_to_img(state)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        return (img, mask,)


class WFC_CustomTemperature:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "starting_temperature": ("INT", {"default": 50, "min": 0, "max": 99}),
                    "min_min_temperature": ("INT", {"default": 0, "min": 0, "max": 99}),
                    "max_min_temperature": ("INT", {"default": 80, "min": 0, "max": 99}),
                },
        }

    RETURN_TYPES = ("WFC_TemperatureConfig",)
    RETURN_NAMES = ("temperature",)
    FUNCTION = "send"
    CATEGORY = "Bmad/WFC"

    def send(self, **kwargs):
        return (kwargs,)


class WFC_CustomValueWeights:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "reverse_depth_w": ("FLOAT", {"default": 1, "min": 0, "max": 1000, "step": .001}),
                    "node_cost_w": ("FLOAT", {"default": 1, "min": 0, "max": 1000, "step": .001}),
                    "prev_state_avg_entropy_w": ("FLOAT", {"default": 0, "min": 0, "max": 1000, "step": .001}),
                },
        }

    RETURN_TYPES = ("WFC_NodeValueConfig",)
    RETURN_NAMES = ("weights",)
    FUNCTION = "send"
    CATEGORY = "Bmad/WFC"

    def send(self, **kwargs):
        return (kwargs,)


class WFC_EmptyState:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "width": ("INT", {"default": 16, "min": 4, "max": 128}),
                    "height": ("INT", {"default": 16, "min": 4, "max": 128}),
                },
        }

    RETURN_TYPES = ("WFC_State",)
    RETURN_NAMES = ("state",)
    FUNCTION = "create"
    CATEGORY = "Bmad/WFC"

    def create(self, width, height):
        return (np.zeros((width, height)),)


class WFC_Filter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "state": ("WFC_State",),
                    "tiles_batch": ("IMAGE",),
                    "invert": ("BOOLEAN", {"default": False}),
                },
        }

    RETURN_TYPES = ("WFC_State",)
    RETURN_NAMES = ("state",)
    FUNCTION = "create"
    CATEGORY = "Bmad/WFC"

    def create(self, state: ndarray, tiles_batch, invert):
        to_filter = [WFC_Sample.tile_to_hash(
            np.clip(255. * tiles_batch[i].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            for i in range(tiles_batch.shape[0])]
        new_state = [t if xor(t in to_filter, invert) else 0 for t in state.flatten()]
        new_state = np.array(new_state).reshape(state.shape)
        return (new_state,)


def generate_single(stop_and_ticker_shm_name, i_kwargs, pid=0): #stop, ticker,
    shm_list = ShareableList(name=stop_and_ticker_shm_name)
    i_kwargs.update({"stop_and_ticker_shm_list": shm_list})
    i_kwargs.update({"pid": pid})
    problem = WFC_Problem(**i_kwargs)
    if problem._number_of_tiles_to_process == 0:
        return i_kwargs["starting_state"]
    try:
        next(best_first_search(problem, graph=True))  # find 1st solution
    except InterruptedError:
        return None
    except StopIteration:
        print("Exhausted all possibilities without finding a complete solution ; or some irregularity occurred.")
    except KeyError:
        print("\33[33m"
              "[wfc_like] WARNING: search exited early due to a key error."
              " This is likely caused by a hashcode collision; if this is a problem"
              ", changing the seed will likely solve it."
              " \33[0m")
        # note that some collisions may not originate an error, and might even allow the return of an invalid state.
        # such occurrences should be very rare though.

    result = problem.get_solution_state()
    return result



class WFC_GenParallel:
    @classmethod
    def INPUT_TYPES(cls):
        gen_types = WFC_GenerateNode.NODE_INPUT_TYPES()
        gen_types["required"]["max_parallel_tasks"] = ("INT", {"default": 4, "min": 1, "max": 32})
        return gen_types

    RETURN_TYPES = ("WFC_State",)
    RETURN_NAMES = ("state",)
    FUNCTION = "gen"
    CATEGORY = "Bmad/WFC"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def gen(self, max_parallel_tasks, custom_temperature_config=None, custom_node_value_config=None, **kwargs):
        from joblib import Parallel, delayed

        max_parallel_tasks = max_parallel_tasks[0]
        ct_len = 0 if custom_temperature_config is None else len(custom_temperature_config)
        cnv_len = 0 if custom_node_value_config is None else len(custom_node_value_config)

        max_len = 0
        for v in kwargs.values():
            if (v_len := len(v)) > max_len:
                max_len = v_len

        # TODO count is also done inside Problem, maybe should use as optional arg to avoid repeating the operation
        ss = kwargs["starting_state"]
        total_tiles_to_proc = sum([i.size - np.count_nonzero(i) for i in ss])
        total_tiles_to_proc += (ss[-1].size - np.count_nonzero(ss[-1])) * (max_len - len(ss))

        items = kwargs.items()
        per_gen_inputs = []
        for i in range(max_len):
            input_i = {item[0]: item[1][min(i, len(item[1]) - 1)] for item in items}
            if ct_len > 0:
                input_i.update(custom_temperature_config[min(i, ct_len - 1)])
            if cnv_len > 0:
                input_i.update(custom_node_value_config[min(i, cnv_len - 1)])
            per_gen_inputs.append(input_i)

        shm_list = ShareableList([False] + [int(0)]*max_len)
        shm_name = shm_list.shm.name

        finished_event = Event()
        pbar: utils.ProgressBar = utils.ProgressBar(total_tiles_to_proc)
        t = Thread(target=waiting_loop, args=(finished_event, pbar, total_tiles_to_proc, shm_name, max_len))
        t.start()

        final_result = Parallel(n_jobs=max_parallel_tasks)(
            delayed(generate_single)(shm_name, per_gen_inputs[i], i) for i in range(max_len))

        terminate_generation(finished_event, shm_list, t)
        return (final_result,)

