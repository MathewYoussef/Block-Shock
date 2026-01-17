### block_shock/src/methods/block_shock.py
## Block-Shock method: complementary masks, semi-structured kernels, output reduce.

#TODO: implement build/forward/backward/step interface


def build(_cfg):
    #TODO: build masks, shard weights, compress to semi-structured
    raise NotImplementedError("Scaffold only: implement build.")


def forward(_state, _x):
    #TODO: run sparse forward on each GPU + all-reduce
    raise NotImplementedError("Scaffold only: implement forward.")


def backward(_state, _loss):
    #TODO: run backward if enabled
    raise NotImplementedError("Scaffold only: implement backward.")


def step(_state):
    #TODO: optimizer step and optional recompress
    raise NotImplementedError("Scaffold only: implement step.")


def teardown(_state):
    #TODO: cleanup if needed
    pass
