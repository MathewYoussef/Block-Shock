### block_shock/src/methods/dense_tp.py
## Dense 2-GPU tensor parallel baseline (row-parallel preferred).

#TODO: implement build/forward/backward/step interface


def build(_cfg):
    #TODO: shard weights and setup TP communication
    raise NotImplementedError("Scaffold only: implement build.")


def forward(_state, _x):
    #TODO: run TP forward + all-reduce or all-gather
    raise NotImplementedError("Scaffold only: implement forward.")


def backward(_state, _loss):
    #TODO: run TP backward if enabled
    raise NotImplementedError("Scaffold only: implement backward.")


def step(_state):
    #TODO: optimizer step if enabled
    raise NotImplementedError("Scaffold only: implement step.")


def teardown(_state):
    #TODO: cleanup if needed
    pass
