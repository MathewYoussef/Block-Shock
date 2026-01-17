### block_shock/src/sparsity/masks.py
## Mask generation and 2:4 validation helpers.

#TODO: generate complementary masks per 4-wide group
#TODO: validate 2-of-4 compliance
#TODO: optional random 2-of-4 mask generator
#TODO: optional magnitude-based selection


def build_masks(_cfg):
    #TODO: build mask tensors based on config
    raise NotImplementedError("Scaffold only: implement mask generation.")


def validate_2of4(_mask) -> None:
    #TODO: assert 2-of-4 structure per group
    raise NotImplementedError("Scaffold only: implement 2:4 check.")
