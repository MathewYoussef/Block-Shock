### block_shock/src/methods/masked_split_dense.py
## Ablation: masked split with dense GEMM (zeros not skipped).

#TODO: implement build/forward/backward/step interface


def build(_cfg):
    #TODO: build masks, shard weights, keep dense
    raise NotImplementedError("Scaffold only: implement build.")


def forward(_state, _x):
    #TODO: run dense forward on masked weights + all-reduce
    raise NotImplementedError("Scaffold only: implement forward.")


def backward(_state, _loss):
    #TODO: run backward if enabled
    raise NotImplementedError("Scaffold only: implement backward.")


def step(_state):
    #TODO: optimizer step if enabled
    raise NotImplementedError("Scaffold only: implement step.")


def teardown(_state):
    #TODO: cleanup if needed
    pass
