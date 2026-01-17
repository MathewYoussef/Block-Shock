### block_shock/src/sparsity/semistructured.py
## Semi-structured compression/decompression and guardrails.

#TODO: convert dense masked -> semi-structured compressed
#TODO: optional decompression for debugging
#TODO: guard supported ops to avoid silent dense fallback


def compress(_weights):
    #TODO: call to_sparse_semi_structured or equivalent
    raise NotImplementedError("Scaffold only: implement compression.")


def decompress(_weights):
    #TODO: convert back to dense for debugging
    raise NotImplementedError("Scaffold only: implement decompression.")
