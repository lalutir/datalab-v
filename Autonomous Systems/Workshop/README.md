# Deep Q-Network (DQN) for Atari Breakout

A working implementation of DeepMind's DQN algorithm trained to play the Atari game Breakout, based on the paper:
> *Playing Atari with Deep Reinforcement Learning* — Mnih et al. (2013)

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Atari ROMs

`AutoROM` was installed in step 2. Run it once to download the game ROMs:

```bash
AutoROM --accept-license
```

### 4. Launch the notebook

```bash
jupyter notebook dqn_breakout.ipynb
```

---

## Project Structure

```
Workshop/
├── dqn_breakout.ipynb   # Main notebook
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── checkpoints/         # Model saves (created at runtime)
```

---

## How the Code Works — Section by Section

### 1. `obs_to_array(obs)`

The Atari environment is wrapped with two gymnasium wrappers that change the observation shape. Depending on the gymnasium version installed:

- gymnasium < 1.0 gives shape `(4, 84, 84)` — channels first
- gymnasium ≥ 1.0 gives shape `(84, 84, 4)` — channels last

PyTorch's `Conv2d` expects channels first: `(batch, channels, H, W)`. This helper always returns a `(4, 84, 84)` float32 array in the range `[0, 1]`, regardless of which gymnasium version is installed.

```python
def obs_to_array(obs) -> np.ndarray:
    arr = np.array(obs, dtype=np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[2] == 4:   # (H, W, 4) → (4, H, W)
        arr = np.transpose(arr, (2, 0, 1))
    return arr
```

---

### 2. `make_env()`

Creates the Atari environment with two preprocessing wrappers:

**`AtariPreprocessing`**
- Converts to grayscale (colour is irrelevant for Breakout)
- Resizes from 210×160 to 84×84 (reduces the number of CNN parameters by ~6×)
- Repeats each action for 4 frames (frame skip), then max-pools the last 2 frames to remove flickering sprites caused by Atari's sprite alternating
- `scale_obs=False` returns `uint8` pixel values (0–255) to save memory

**`FrameStack(env, 4)`**
Stacks 4 consecutive frames into a single observation. A single frame is ambiguous — you cannot tell whether the ball is moving left or right, or how fast it is moving. By stacking 4 frames, the CNN can infer velocity and direction from the difference between frames.

---

### 3. Hyperparameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `BUFFER_SIZE` | 100 000 | How many past transitions to remember. Too small → overfits to recent experience. Too large → requires too much RAM. |
| `BATCH_SIZE` | 32 | Transitions sampled per gradient update. Larger = more stable gradients but slower per step. |
| `GAMMA` | 0.99 | Discount factor. 0.99 means the agent cares about rewards up to ~100 steps in the future. |
| `LEARNING_RATE` | 1e-4 | Adam step size. Too high → training diverges. Too low → learns very slowly. |
| `NUM_FRAMES` | 1 000 000 | Total environment steps. Full convergence needs ~5–10 million. |
| `MIN_REPLAY_SIZE` | 10 000 | Steps of random play before training starts (buffer warm-up). |
| `TRAIN_FREQ` | 4 | Train every 4 environment steps. Avoids over-training on the same transitions. |
| `EPSILON_START` | 1.0 | Agent starts completely random. |
| `EPSILON_END` | 0.1 | Always keeps 10% random exploration even when fully trained. |
| `EPSILON_DECAY_FRAMES` | 500 000 | Frames over which ε decays linearly from 1.0 to 0.1. |
| `TARGET_UPDATE_FREQ` | 1 000 | How often to hard-copy the policy network into the target network. |

---

### 4. `ReplayBuffer`

Stores past game transitions in a circular buffer (oldest entries are discarded when full).

**Why not train on consecutive frames?**  
Frames from the same game moment are highly correlated — they show nearly identical situations. Training on them in sequence would cause the network to overfit to recent experience and forget earlier knowledge (a problem called *catastrophic forgetting*). Randomly sampling from a large buffer breaks these correlations.

**Memory efficiency**  
Each 84×84 `uint8` frame costs 7 056 bytes. As `float32` it would cost 28 224 bytes — 4× more. By storing as `uint8` and casting to `float32` only when sampled, a buffer of 100 000 transitions uses roughly 2.7 GB instead of 10.8 GB.

```
push() → converts obs to (4,84,84) uint8 → stores tuple
sample() → random mini-batch → returns float32 normalised to [0,1]
```

---

### 5. `DQN` (the neural network)

A convolutional neural network that takes 4 stacked frames and outputs one Q-value per action.

**Q-value**: the expected total future reward when taking action `a` in state `s`, then following the current policy thereafter.

```
Input  (batch, 4, 84, 84)
  │
  ├─ Conv2d(4→32,  8×8, stride 4)   → (batch, 32, 20, 20)
  ├─ ReLU
  ├─ Conv2d(32→64, 4×4, stride 2)   → (batch, 64,  9,  9)
  ├─ ReLU
  ├─ Conv2d(64→64, 3×3, stride 1)   → (batch, 64,  7,  7)
  ├─ ReLU
  ├─ Flatten                         → (batch, 3136)
  ├─ Linear(3136 → 512) + ReLU
  └─ Linear(512 → 4)                 → (batch, 4)  ← one Q-value per action
```

Large strides in the first two conv layers quickly reduce spatial resolution while extracting features, keeping the parameter count manageable (~1.7 M parameters).

---

### 6. `DQNAgent`

Combines the policy network, target network, replay buffer, and training logic.

#### Epsilon-Greedy Exploration

```
if random() < epsilon:
    action = random choice          ← EXPLORE
else:
    action = argmax Q(state, a)    ← EXPLOIT
```

ε decays linearly from 1.0 (fully random) to 0.1 (mostly greedy) over `EPSILON_DECAY_FRAMES` steps. The minimum of 0.1 ensures the agent never completely stops exploring.

#### `train_step()` — The Bellman Update

1. Sample a random batch of 32 transitions from the replay buffer
2. Pass states through the **policy network** → get current Q-values
3. Use `.gather()` to select only the Q-value for the action that was taken
4. Pass next-states through the **target network** → get max Q-value for next state
5. Compute the **Bellman target**: `r + γ · max Q_target(s', a') · (1 − done)`
6. Compute **Huber loss** between current and target Q-values
7. Backpropagate, clip gradients (max norm 10), update weights

#### Why Two Networks?

If the same network is used for both selecting actions and computing targets, the targets shift every time the weights update. This creates a feedback loop that often causes training to diverge.

The **target network** is a frozen snapshot of the policy network, copied every `TARGET_UPDATE_FREQ` steps. It provides stable, slowly-changing targets — analogous to having a fixed set of labels in supervised learning.

---

### 7. `train()` — The Training Loop

The main loop runs for `NUM_FRAMES` environment steps:

```
for each frame:
    1. Select action (epsilon-greedy)
    2. Step environment → (next_state, reward, terminated, truncated, info)
    3. Detect life loss → treat as terminal for training
    4. Clip reward to [-1, 1]
    5. Store transition in replay buffer
    6. If buffer ready: train_step() + update target + checkpoint
```

**Life loss as terminal**  
Breakout gives the player 5 lives. Without special handling, the agent treats life loss as unimportant — it knows more balls will follow. We detect when the `lives` count in `info` drops and set `done = True` in the replay buffer. This signals to the network that losing a life is bad, encouraging the agent to protect its lives. The environment is only truly reset (`env.reset()`) when all lives are gone.

**Reward clipping**  
Clipping rewards to `[−1, +1]` keeps the gradient scale consistent regardless of the game's actual scoring scheme. Without clipping, games with scores in the thousands would require much lower learning rates than games with binary rewards.

---

### 8. Plotting & Evaluation

**`plot_results()`** shows:
- Per-episode rewards (raw, noisy) with a smoothed moving average to reveal the trend
- Huber loss per training step (smoothed) to diagnose convergence

**`evaluate()`** runs the agent greedily (ε = 0) for N episodes and reports the total reward. A well-trained agent (after ~5 M frames) typically scores 20–400 points per episode.

---

## Training Time

| Hardware | Frames/second | Time for 1 M frames |
|----------|--------------|---------------------|
| GPU (RTX 3080) | ~800–1200 | ~15–20 minutes |
| GPU (T4, Colab) | ~400–600 | ~30–40 minutes |
| CPU only | ~80–150 | ~2–3 hours |

For a quick smoke test, set `NUM_FRAMES = 50_000` — training finishes in a few minutes and you can verify that all code runs correctly before committing to a full training run.

---

## Common Issues

**`ale-py` or ROM not found**  
Make sure `AutoROM --accept-license` was run after installation. If you see `ROM not found`, try reinstalling: `pip install ale-py AutoROM[accept-rom-license]` then `AutoROM --accept-license`.

**CUDA out of memory**  
Reduce `BUFFER_SIZE` to 50 000. The replay buffer is stored in CPU RAM, but intermediate tensors live on the GPU.

**Training does not start / stays at reward 0**  
Normal for the first `MIN_REPLAY_SIZE` frames — the agent plays randomly until the buffer is full. Once training starts, reward will remain low until ε drops below ~0.3 (after ~400 000 frames with default settings).

**`weights_only=True` warning / error on `torch.load`**  
This is a PyTorch ≥ 2.0 security default. The `load()` method already passes `weights_only=True`. If you encounter an error, you are likely loading a checkpoint saved by an older version — pass `weights_only=False` temporarily.
