import os

shared_params = [
    "dataset=WisconsinXAllAndHopSignDemo",  # runs reasonably quickly; for full (slow) set of benchmarks, use WisconsinXAllAndHopSign
    "seed=0",
]

hparams = {
    "graphany": [
        "prev_ckpt=checkpoints/graph_any_wisconsin.pt",
        "seed=0",
        "total_steps=0",
    ],  # X+L1+L2(+H1+H2)
    "graphany_extended": [
        "feat_chn=X+L1+L2+L3+L4+H1+H2",
        "pred_chn=X+L1+L2+L3+L4",
        "entropy=1",
        "attn_temp=5",
        "n_hidden=128",
        "n_mlp_layer=3",
        "total_steps=1500",
        "seed=0",
    ],  # X+L1+L2*+L3+L4*(+H1+H2)
    "N1-4": [
        "feat_chn=X+N1+N2+N3+N4",
        "pred_chn=X+N1+N2+N3+N4",
        "dataset=WisconsinXAllAndHopSign",
        "entropy=2",
        "attn_temp=5",
        "n_hidden=64",
        "n_mlp_layer=3,",
        "total_steps=1500",
        "seed=0",
    ],  # X+A_1+A_2+A_3+A_4
    "binned_hops": [
        "feat_chn=X+N1+N2+N3d+NdR",
        "pred_chn=X+N1+N2+N3d+NdR",
        "entropy=2",
        "attn_temp=5",
        "n_hidden=128",
        "n_mlp_layer=3",
        "total_steps=1000",
    ],  # precise hops with binned medium- and long-range bins; see paper for details
    "heat_kernel": [
        "feat_chn=X+N1+DS+DM+DL",
        "pred_chn=X+N1+DS+DM+DL",
        "entropy=2",
        "attn_temp=5",
        "n_hidden=64",
        "n_mlp_layer=2",
        "total_steps=1500",
    ],  # X+L1+ heat kernel operators (short, medium, long); see paper for details
}

mode = "graphany"

if os.getcwd().endswith("notebooks"):
    os.chdir("..")

run_cmd = f"python graphany/run.py {' '.join(shared_params + hparams[mode])}"
print(run_cmd)
os.system(run_cmd)
