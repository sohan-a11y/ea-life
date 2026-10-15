"""Compact neural latent world model with bounded replay and multi-step imagination.

Pure Python by design: no external ML runtime is needed in the isolated life container.
This is a small research prototype, not a production-scale learned simulator.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

FEATURE_DIM = 14
LATENT_DIM = 8
HIDDEN_DIM = 24
ACTION_COUNT = 9
REWARD_SCALE = 8.0
REPLAY_CAPACITY = 2048
WARMUP_TRANSITIONS = 48
TRAINING_UPDATES_PER_TRANSITION = 2


def _tanh(value: float) -> float:
    return math.tanh(max(-12.0, min(12.0, value)))


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _matrix(rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    scale = math.sqrt(6.0 / (rows + cols))
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _vector(rng: random.Random, size: int) -> list[float]:
    return [rng.uniform(-0.01, 0.01) for _ in range(size)]


def _dense(weights: Sequence[Sequence[float]], bias: Sequence[float],
           values: Sequence[float]) -> list[float]:
    return [sum(w * x for w, x in zip(row, values)) + b
            for row, b in zip(weights, bias)]


def _as_float_vector(values: Sequence[float], length: int) -> list[float]:
    if len(values) != length:
        raise ValueError("expected vector length %d, got %d" % (length, len(values)))
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("model inputs must be finite")
    return [max(-1.0, min(1.0, value)) for value in result]


class NeuralLatentWorldModel:
    """Autoencoder + latent transition/reward MLP + bounded shared experience pool.

    Observations are normalized feature vectors, encoded to a learned latent vector.
    A learned dynamics network predicts the next latent vector and immediate reward.
    Beam search chains those predictions into short imagined action sequences.
    """

    def __init__(self, seed: int = 20261016, learning_rate: float = 0.008,
                 replay_capacity: int = REPLAY_CAPACITY):
        if not 16 <= replay_capacity <= 16384:
            raise ValueError("replay capacity must be between 16 and 16384")
        self.rng = random.Random(seed)
        self.learning_rate = float(learning_rate)
        self.replay_capacity = replay_capacity
        self.replay: list[dict[str, Any]] = []
        self.encoder_w = _matrix(self.rng, LATENT_DIM, FEATURE_DIM)
        self.encoder_b = _vector(self.rng, LATENT_DIM)
        self.decoder_w = _matrix(self.rng, FEATURE_DIM, LATENT_DIM)
        self.decoder_b = _vector(self.rng, FEATURE_DIM)
        transition_input = LATENT_DIM + ACTION_COUNT
        self.dynamics_w1 = _matrix(self.rng, HIDDEN_DIM, transition_input)
        self.dynamics_b1 = _vector(self.rng, HIDDEN_DIM)
        self.dynamics_w2 = _matrix(self.rng, LATENT_DIM + 1, HIDDEN_DIM)
        self.dynamics_b2 = _vector(self.rng, LATENT_DIM + 1)
        self.transitions_seen = 0
        self.training_updates = 0
        self.transition_loss_ema = 0.0
        self.reconstruction_loss_ema = 0.0
        self.reward_mae_ema = 0.0
        self.imagination_calls = 0
        self.imagination_predictions = 0
        self.last_plan_return = 0.0
        self.last_plan_horizon = 0

    def encode(self, observation: Sequence[float]) -> list[float]:
        x = _as_float_vector(observation, FEATURE_DIM)
        return [_tanh(value) for value in _dense(self.encoder_w, self.encoder_b, x)]

    def decode(self, latent: Sequence[float]) -> list[float]:
        z = _as_float_vector(latent, LATENT_DIM)
        return [_sigmoid(value) for value in _dense(self.decoder_w, self.decoder_b, z)]

    def reconstruct(self, observation: Sequence[float]) -> list[float]:
        return self.decode(self.encode(observation))

    @staticmethod
    def _dynamics_input(latent: Sequence[float], action: int) -> list[float]:
        if not 0 <= int(action) < ACTION_COUNT:
            raise ValueError("action index outside model action space")
        return list(latent) + [1.0 if i == int(action) else 0.0 for i in range(ACTION_COUNT)]

    def predict(self, latent: Sequence[float], action: int) -> tuple[list[float], float]:
        """Predict next latent state and immediate reward without changing model state."""
        z = _as_float_vector(latent, LATENT_DIM)
        x = self._dynamics_input(z, action)
        hidden = [_tanh(value) for value in _dense(self.dynamics_w1, self.dynamics_b1, x)]
        out = [_tanh(value) for value in _dense(self.dynamics_w2, self.dynamics_b2, hidden)]
        next_latent = out[:LATENT_DIM]
        reward = max(-1.0, min(1.0, out[-1])) * REWARD_SCALE
        return next_latent, reward

    def _train_autoencoder(self, observation: Sequence[float]) -> float:
        x = _as_float_vector(observation, FEATURE_DIM)
        z_pre = _dense(self.encoder_w, self.encoder_b, x)
        z = [_tanh(value) for value in z_pre]
        y_pre = _dense(self.decoder_w, self.decoder_b, z)
        y = [_sigmoid(value) for value in y_pre]
        error = [y[i] - x[i] for i in range(FEATURE_DIM)]
        loss = sum(value * value for value in error) / FEATURE_DIM
        grad_y_pre = [2.0 * error[i] / FEATURE_DIM * y[i] * (1.0 - y[i])
                      for i in range(FEATURE_DIM)]
        grad_z = [0.0 for _ in range(LATENT_DIM)]
        # Backpropagate through the decoder before mutating its weights.
        for i in range(FEATURE_DIM):
            for j in range(LATENT_DIM):
                grad_z[j] += self.decoder_w[i][j] * grad_y_pre[i]
        lr = self.learning_rate
        for i in range(FEATURE_DIM):
            for j in range(LATENT_DIM):
                self.decoder_w[i][j] -= lr * grad_y_pre[i] * z[j]
            self.decoder_b[i] -= lr * grad_y_pre[i]
        grad_z_pre = [grad_z[j] * (1.0 - z[j] * z[j]) for j in range(LATENT_DIM)]
        for j in range(LATENT_DIM):
            for k in range(FEATURE_DIM):
                self.encoder_w[j][k] -= lr * grad_z_pre[j] * x[k]
            self.encoder_b[j] -= lr * grad_z_pre[j]
        self.reconstruction_loss_ema = (0.97 * self.reconstruction_loss_ema
                                         + 0.03 * loss)
        return loss

    def _train_transition(self, observation: Sequence[float], action: int,
                          next_observation: Sequence[float], reward: float) -> float:
        state_latent = self.encode(observation)
        target_latent = self.encode(next_observation)
        model_input = self._dynamics_input(state_latent, action)
        pre1 = _dense(self.dynamics_w1, self.dynamics_b1, model_input)
        hidden = [_tanh(value) for value in pre1]
        pre2 = _dense(self.dynamics_w2, self.dynamics_b2, hidden)
        outputs = [_tanh(value) for value in pre2]
        target_reward = max(-1.0, min(1.0, float(reward) / REWARD_SCALE))
        target = list(target_latent) + [target_reward]
        latent_error = [outputs[i] - target[i] for i in range(LATENT_DIM)]
        reward_error = outputs[-1] - target[-1]
        latent_loss = sum(value * value for value in latent_error) / LATENT_DIM
        loss = latent_loss + 0.5 * reward_error * reward_error
        grad_out_pre = []
        for i in range(LATENT_DIM):
            grad_out_pre.append(2.0 * latent_error[i] / LATENT_DIM
                                * (1.0 - outputs[i] * outputs[i]))
        grad_out_pre.append(reward_error * (1.0 - outputs[-1] * outputs[-1]))
        grad_hidden = [0.0 for _ in range(HIDDEN_DIM)]
        for i in range(LATENT_DIM + 1):
            for j in range(HIDDEN_DIM):
                grad_hidden[j] += self.dynamics_w2[i][j] * grad_out_pre[i]
        lr = self.learning_rate
        for i in range(LATENT_DIM + 1):
            for j in range(HIDDEN_DIM):
                self.dynamics_w2[i][j] -= lr * grad_out_pre[i] * hidden[j]
            self.dynamics_b2[i] -= lr * grad_out_pre[i]
        grad_hidden_pre = [grad_hidden[j] * (1.0 - hidden[j] * hidden[j])
                           for j in range(HIDDEN_DIM)]
        for i in range(HIDDEN_DIM):
            for j in range(len(model_input)):
                self.dynamics_w1[i][j] -= lr * grad_hidden_pre[i] * model_input[j]
            self.dynamics_b1[i] -= lr * grad_hidden_pre[i]
        self.training_updates += 1
        self.transition_loss_ema = (0.97 * self.transition_loss_ema + 0.03 * loss)
        # The reward MAE is reported in environment reward units.
        self.reward_mae_ema = 0.97 * self.reward_mae_ema + 0.03 * abs(
            outputs[-1] * REWARD_SCALE - float(reward))
        return loss

    def _train_sample(self, sample: dict[str, Any]) -> None:
        self._train_transition(sample["obs"], int(sample["action"]),
                               sample["next_obs"], float(sample["reward"]))
        self._train_autoencoder(sample["obs"])
        self._train_autoencoder(sample["next_obs"])

    def observe_transition(self, observation: Sequence[float], action: int,
                           next_observation: Sequence[float], reward: float) -> None:
        """Add one real transition and train online from it plus a replay sample."""
        obs = _as_float_vector(observation, FEATURE_DIM)
        nxt = _as_float_vector(next_observation, FEATURE_DIM)
        if not 0 <= int(action) < ACTION_COUNT:
            raise ValueError("action index outside model action space")
        if not math.isfinite(float(reward)):
            raise ValueError("reward must be finite")
        sample = {"obs": obs, "action": int(action), "next_obs": nxt,
                  "reward": float(reward)}
        self.replay.append(sample)
        if len(self.replay) > self.replay_capacity:
            del self.replay[0]
        self.transitions_seen += 1
        self._train_sample(sample)
        if len(self.replay) > 1:
            self._train_sample(self.rng.choice(self.replay))

    @property
    def ready(self) -> bool:
        return self.transitions_seen >= WARMUP_TRANSITIONS and self.training_updates >= WARMUP_TRANSITIONS

    def imagine(self, observation: Sequence[float], horizon: int = 4,
                beam_width: int = 3, gamma: float = 0.9,
                legal_first_actions: Sequence[int] | None = None) -> dict[str, Any] | None:
        """Beam-search action sequences in latent space and return the best first action."""
        if not self.ready:
            return None
        if not 1 <= horizon <= 12:
            raise ValueError("imagination horizon must be between 1 and 12")
        if not 1 <= beam_width <= 8:
            raise ValueError("imagination beam width must be between 1 and 8")
        initial = self.encode(observation)
        first_actions = list(range(ACTION_COUNT)) if legal_first_actions is None else sorted(set(int(a) for a in legal_first_actions))
        if not first_actions or any(a < 0 or a >= ACTION_COUNT for a in first_actions):
            raise ValueError("legal first actions must contain valid action indices")
        # Node tuple: latent state, discounted score, first real action, imagined path.
        beam: list[tuple[list[float], float, int | None, tuple[int, ...]]] = [
            (initial, 0.0, None, ())
        ]
        self.imagination_calls += 1
        evaluated = 0
        for depth in range(horizon):
            candidates: list[tuple[list[float], float, int | None, tuple[int, ...]]] = []
            for latent, total, first_action, path in beam:
                actions = first_actions if depth == 0 else range(ACTION_COUNT)
                for action in actions:
                    next_latent, reward = self.predict(latent, action)
                    evaluated += 1
                    first = action if first_action is None else first_action
                    score = total + (gamma ** depth) * reward
                    candidates.append((next_latent, score, first, path + (action,)))
            candidates.sort(key=lambda node: node[1], reverse=True)
            # Retain a bounded beam, preferring distinct first actions to reduce collapse.
            beam = []
            seen_first: set[int] = set()
            for node in candidates:
                if node[2] not in seen_first or len(seen_first) >= min(beam_width, len(candidates)):
                    beam.append(node)
                    if node[2] is not None:
                        seen_first.add(node[2])
                    if len(beam) >= beam_width:
                        break
            if not beam:
                return None
        best = max(beam, key=lambda node: node[1])
        self.imagination_predictions += evaluated
        self.last_plan_return = float(best[1])
        self.last_plan_horizon = horizon
        return {"action": int(best[2]), "return": float(best[1]),
                "path": list(best[3]), "horizon": horizon,
                "beam_width": beam_width, "predictions": evaluated}

    def metrics(self) -> dict[str, Any]:
        return {
            "latent_model_ready": self.ready,
            "latent_transitions_seen": self.transitions_seen,
            "latent_training_updates": self.training_updates,
            "latent_replay_size": len(self.replay),
            "latent_reconstruction_loss": round(self.reconstruction_loss_ema, 5),
            "latent_transition_loss": round(self.transition_loss_ema, 5),
            "latent_reward_mae": round(self.reward_mae_ema, 4),
            "latent_imagination_calls": self.imagination_calls,
            "latent_imagination_predictions": self.imagination_predictions,
            "latent_last_plan_return": round(self.last_plan_return, 4),
            "latent_last_plan_horizon": self.last_plan_horizon,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "learning_rate": self.learning_rate,
            "replay_capacity": self.replay_capacity,
            "replay": self.replay,
            "encoder_w": self.encoder_w, "encoder_b": self.encoder_b,
            "decoder_w": self.decoder_w, "decoder_b": self.decoder_b,
            "dynamics_w1": self.dynamics_w1, "dynamics_b1": self.dynamics_b1,
            "dynamics_w2": self.dynamics_w2, "dynamics_b2": self.dynamics_b2,
            "transitions_seen": self.transitions_seen,
            "training_updates": self.training_updates,
            "transition_loss_ema": self.transition_loss_ema,
            "reconstruction_loss_ema": self.reconstruction_loss_ema,
            "reward_mae_ema": self.reward_mae_ema,
            "imagination_calls": self.imagination_calls,
            "imagination_predictions": self.imagination_predictions,
            "last_plan_return": self.last_plan_return,
            "last_plan_horizon": self.last_plan_horizon,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported latent model checkpoint version")
        for name in ("encoder_w", "encoder_b", "decoder_w", "decoder_b",
                     "dynamics_w1", "dynamics_b1", "dynamics_w2", "dynamics_b2"):
            if name not in state:
                raise ValueError("latent model checkpoint is missing " + name)
            setattr(self, name, state[name])
        self.learning_rate = float(state["learning_rate"])
        self.replay_capacity = int(state["replay_capacity"])
        self.replay = list(state.get("replay", []))[-self.replay_capacity:]
        for name in ("transitions_seen", "training_updates", "imagination_calls",
                     "imagination_predictions", "last_plan_horizon"):
            setattr(self, name, int(state.get(name, 0)))
        for name in ("transition_loss_ema", "reconstruction_loss_ema", "reward_mae_ema",
                     "last_plan_return"):
            setattr(self, name, float(state.get(name, 0.0)))
        self.rng.setstate(_tupleify(state["rng_state"]))

    @staticmethod
    def _validate_observation(observation: Sequence[float]) -> list[float]:
        return _as_float_vector(observation, FEATURE_DIM)


def _tupleify(value: Any) -> Any:
    return tuple(_tupleify(item) for item in value) if isinstance(value, list) else value
