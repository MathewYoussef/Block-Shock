### block_shock/src/methods/dense_single.py
## Dense single-GPU baseline.

#TODO: implement build/forward/backward/step interface


def build(_cfg):
    #TODO: allocate dense weights on one GPU
    raise NotImplementedError("Scaffold only: implement build.")


def forward(_state, _x):
    #TODO: run dense forward
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
