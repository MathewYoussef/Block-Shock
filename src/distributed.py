### block_shock/src/distributed.py
## Distributed helpers: process group init, device assignment, all-reduce.

#TODO: init process group (nccl)
#TODO: assign local ranks and devices
#TODO: implement all-reduce wrappers with timing hooks


def init_distributed(_cfg) -> None:
    #TODO: initialize torch.distributed from config
    raise NotImplementedError("Scaffold only: implement distributed setup.")
