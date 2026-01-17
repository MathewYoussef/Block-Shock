### block_shock/src/workloads.py
## Synthetic input and loss generation for benchmarks.

#TODO: parameterize distributions (normal, activation-like, fixed seed)
#TODO: build inputs on correct device/dtype
#TODO: build loss function for backward phases


def build_inputs(_cfg):
    #TODO: generate X based on workload config
    raise NotImplementedError("Scaffold only: implement workloads.")


def build_loss(_cfg):
    #TODO: return loss function or reduction
    raise NotImplementedError("Scaffold only: implement loss selection.")
