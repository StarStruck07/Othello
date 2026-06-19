# Othello Engine — RL & MCTS

A single Othello codebase combining two agents that share one game engine and
one supervised pre-training pipeline:

- **RL** — a Dueling-DQN trained with self-play and a snapshot opponent pool.
- **MCTS** — an AlphaZero-style policy/value network trained with MCTS self-play.

Both can be benchmarked against the [Edax](https://github.com/abulmo/edax) engine.

## Layout

```
othello/
├── common/                  # shared, model-agnostic — used by BOTH agents
│   ├── othello.py             # board rules + Tk visualiser
│   ├── game.py                # Environment / state encoding (3×8×8 planes)
│   ├── RandomOpp.py           # evaluate_against_random
│   ├── superVisionClassifier.py   # supervised pre-training (--variant rl|mcts)
│   ├── plot_against_snapshot.py
│   └── othello_dataset.csv
│
├── rl/                      # Dueling-DQN agent
│   ├── model.py               # ResNet trunk + value & advantage heads -> Q(s,a)
│   ├── agent.py               # DQN training loop, replay buffer, snapshots
│   ├── visualize_q_values.py
│   └── plot_training_logs.py
│
├── mcts/                    # AlphaZero-style agent
│   ├── model.py               # ResNet trunk + policy & value heads
│   ├── mcts.py                # PUCT search
│   ├── agent.py               # self-play training loop
│   ├── pit_checkpoints.py     # round-robin between iter checkpoints
│   ├── plot_training_logs.py
│   └── *.pt, training_logs.json, *.png   # trained checkpoints & artifacts
│
└── edax/                    # benchmarking vs the Edax engine (BOTH agents)
    ├── edax_engine.py         # shared Edax driver (protocol, sync-safety, eval, plot)
    ├── vs_edax_rl.py          # RL front-end   (masked Q-value argmax)
    ├── vs_edax_mcts.py        # MCTS front-end (raw policy, or --sims N for search)
    └── debug_edax.py          # one verbose game; --variant rl|mcts
```

### What is shared vs. variant-specific

The two models differ only in their **heads** and how a move is chosen:

| | trunk (ResNet) | head | move selection |
|---|---|---|---|
| RL | shared arch | value + advantage → Q-values | mask illegal, argmax Q |
| MCTS | shared arch | policy + value | policy argmax, or PUCT search |

The supervised classifier head (`supervised=True`) is **identical** in both
models, so one pre-training script produces a warm-start usable by either agent.

## How to run

Each script adds `../common` to `sys.path` automatically and resolves `model`
(and `mcts`) from its own folder, so just run files from inside their directory:

```bash
# Supervised pre-training (choose the architecture to warm-start)
cd common
python superVisionClassifier.py --variant rl       # -> best_supervised.pt
python superVisionClassifier.py --variant mcts

# RL training
cd ../rl    && python agent.py

# MCTS training
cd ../mcts  && python agent.py
python pit_checkpoints.py                            # compare iter checkpoints

# Benchmark vs Edax  (needs: pip install pexpect, and an Edax binary + data/eval.dat)
cd ../edax
python vs_edax_rl.py   --binary /path/to/mEdax-native --checkpoint ../rl/othello_dqn_model.pt --levels 1 3 5 --games 20
python vs_edax_mcts.py --binary /path/to/mEdax-native --checkpoint ../mcts/othello_az_iter100.pt --sims 100
python debug_edax.py   --variant mcts --binary /path/to/mEdax-native --checkpoint ../mcts/othello_az_iter100.pt --level 5
```

> The RL `agent.py` expects a supervised warm-start named `othello_supervised.pt`
> in its folder; the MCTS `agent.py` expects `othello_supervised_final.pt`. Both
> are produced by `common/superVisionClassifier.py` (run with the matching
> `--variant`) — rename/copy the output into the variant folder as needed.
