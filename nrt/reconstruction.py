import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from nrt.helpers import encode


class Decoder(nn.Module):
    """MLP that reconstructs (x, y) coordinates from neural activations."""
    hidden_sizes: tuple = (64, 64)
    out_dim: int = 2

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


def reconstruction_loss(params, model, V, coords):
    pred = model.apply(params, V)
    return jnp.mean(jnp.sum((pred - coords) ** 2, axis=-1))


@jax.jit(static_argnames=("model", "optimizer"))
def train_step(params, opt_state, model, optimizer, V, coords):
    loss, grads = jax.value_and_grad(reconstruction_loss)(params, model, V, coords)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_reconstruction_from_encoder(key, g0, om, S, dataloader, num_batches=None,
                                      hidden_sizes=(64, 64), lr=1e-3):
    """Encode trajectory data with a trained plane sequential model, then train a decoder.

    Loads batches from the dataloader, encodes all positions using the sequential
    plane encoder, and trains a Decoder MLP to reconstruct (x, y) from the
    resulting neural activations.

    Args:
        key: JAX PRNG key.
        g0: Trained activity at origin, shape [D].
        om: Trained frequencies, shape [M, 2].
        S: Trained change of basis matrix, shape [D, D].
        dataloader: A TrajectoryDataset from nrt.data.
        num_batches: Number of batches to encode. None uses all batches.
        hidden_sizes: Decoder hidden layer widths.
        lr: Learning rate.

    Returns:
        (trained_params, model, losses) where model is the Decoder instance.
    """

    if num_batches is None:
        num_batches = dataloader.num_batches

    # Flax needs a forward pass to trace layer shapes and allocate weights.
    # A single zero vector with the right input dim (D = 2*M+1) suffices.
    D = g0.shape[0]
    model = Decoder(hidden_sizes=hidden_sizes, out_dim=2)
    params = model.init(key, jnp.zeros((1, D)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    losses = []
    for batch_idx in range(num_batches):
        # batch shape: [batch_size, sequence_length, 2]
        batch = np.array(dataloader.get_batch(batch_idx))
        phi = jnp.array(batch)

        # Encode: returns [B, L, D]
        V = encode(g0, om, S, phi)

        # Flatten batch and sequence dims
        B, L, D = V.shape

        V = V.reshape(B * L, D)
        batch = jnp.asarray(batch).reshape(B * L, 2)
        params, opt_state, loss = train_step(params, opt_state, model, optimizer, V, batch)
        losses.append(loss)

    return params, model, jnp.array(losses)
