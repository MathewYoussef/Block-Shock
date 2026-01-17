### block_shock/src/main.py
## Entrypoint for the experiment runner.
# Loads YAML config stack, initializes distributed, iterates methods, runs phases, logs results.

#TODO: add CLI args for config paths and overrides
#TODO: implement YAML merge stack (base + phase + method + workload + hardware + sweeps)
#TODO: initialize distributed if requested
#TODO: build workload inputs
#TODO: iterate methods and run the phase pipeline
#TODO: write JSONL results per run


def main() -> None:
    #TODO: parse args, load config, run experiment
    raise NotImplementedError("Scaffold only: implement main entrypoint.")


if __name__ == "__main__":
    main()
