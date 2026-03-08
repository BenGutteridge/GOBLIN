import os

# CityNetworks are large (114K–568K nodes). The default max_khops=10 would
# allocate an N×N APSPD matrix and OOM. Override to use sampled BFS from a
# small number of source nodes instead. The sampled APSPD is only used to
# compute dist_median/dist_mean, which are unused by the standard
# X+L1+L2+H1+H2 channels, so this does not affect accuracy.
large_graph_params = [
    "max_khops=null",
    "max_output_nodes_samples=10",
]

extra_hparams = [
    "total_steps=0",
    "feat_chn=X+L1+L2+H1+H2",
    "pred_chn=X+L1+L2",
]

SEED = 0

hparams = {
    "graphany_wisconsin": {
        "dataset": "WisconsinXAllCityNetworks",
        "extra": [
            "prev_ckpt=checkpoints/graph_any_wisconsin.pt",
            f"seed={SEED}",
        ] + extra_hparams,
    },
    "graphany_cora": {
        "dataset": "CoraXAllCityNetworks",
        "extra": [
            "prev_ckpt=checkpoints/graph_any_cora.pt",
            f"seed={SEED}",
        ] + extra_hparams,
    },
}

mode = "graphany_cora"

if os.getcwd().endswith("notebooks"):
    os.chdir("..")

cfg = hparams[mode]
params = [f"dataset={cfg['dataset']}", "seed=0"] + large_graph_params + cfg["extra"]
run_cmd = f"python graphany/run.py {' '.join(params)}"
print(run_cmd)
os.system(run_cmd)
