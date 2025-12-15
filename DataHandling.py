# -*- coding: utf-8 -*-
import jax.numpy as jnp
import numpy as np
import os
import json
import pickle
from functools import lru_cache


class TrajectoryDataset:
    """
    Efficient dataset loader for pre-generated trajectory data with LRU caching.
    """
    def __init__(self, dataset_path, cache_size=128):
        """
        Args:
            dataset_path: Path to the directory containing the dataset
            cache_size: Number of batch files to keep in memory (default: 128)
        """
        self.dataset_path = dataset_path

        # Load metadata
        with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        self.num_batches = self.metadata['num_batches']
        self.batch_size = self.metadata['batch_size']
        self.sequence_length = self.metadata['sequence_length']

        # Create cached load function with specified cache size
        self._load_batch = lru_cache(maxsize=cache_size)(self._load_batch_uncached)

    def __len__(self):
        return self.num_batches * self.batch_size

    def _load_batch_uncached(self, batch_idx):
        """Load a batch file from disk (uncached)"""
        batch_path = os.path.join(self.dataset_path, f'batch_{batch_idx:05d}.pkl')
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f)
        return batch

    def __getitem__(self, idx):
        # Determine which batch file and which sample within that batch
        batch_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size

        # Load the batch (from cache if available)
        batch = self._load_batch(batch_idx)

        # Extract the specific trajectory (shape: [sequence_length, 2])
        trajectory = batch[:, sample_idx, :]

        return trajectory

    def get_batch(self, batch_idx):
        """Get an entire batch file"""
        return self._load_batch(batch_idx)

    def clear_cache(self):
        """Clear the cache to free memory"""
        self._load_batch.cache_clear()

    def cache_info(self):
        """Get cache statistics"""
        return self._load_batch.cache_info()


class TrajectoryGenerator(object):

    def __init__(self, periodic=False):
        self.periodic = periodic

    def avoid_wall(self, position, hd, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:, 0]
        y = position[:, 1]
        dists = [box_width / 2 - x, box_height / 2 - y, box_width / 2 + x, box_height / 2 + y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(self, box_width, box_height, batch_size, sequence_length):
        '''
        Generate a random walk in a rectangular box.

        Returns:
            positions: numpy array of shape [batch_size, sequence_length, 2]
                      Contains (x, y) coordinates for each timestep
        '''
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, sequence_length, 2])
        head_dir = np.zeros([batch_size, sequence_length])

        # Random initial positions and heading
        position[:, 0, 0] = np.random.uniform(-box_width / 2, box_width / 2, batch_size)
        position[:, 0, 1] = np.random.uniform(-box_height / 2, box_height / 2, batch_size)
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, sequence_length - 1])
        random_vel = np.random.rayleigh(b, [batch_size, sequence_length - 1])

        for t in range(sequence_length - 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            if not self.periodic:
                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(position[:, t], head_dir[:, t], box_width, box_height)
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity = v * dt
            update = velocity[:, None] * np.stack([np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        # Periodic boundaries
        if self.periodic:
            position[:, :, 0] = np.mod(position[:, :, 0] + box_width / 2, box_width) - box_width / 2
            position[:, :, 1] = np.mod(position[:, :, 1] + box_height / 2, box_height) - box_height / 2

        return position

    def get_generator(self, batch_size=32, box_width=2, box_height=2, sequence_length=50):
        '''
        Returns a generator that yields batches of trajectories

        Yields:
            positions: jax.numpy array of shape [sequence_length, batch_size, 2]
        '''
        while True:
            positions = self.generate_trajectory(box_width, box_height, batch_size, sequence_length)
            positions = jnp.array(positions).transpose(1, 0, 2)
            yield positions

    def get_test_batch(self, batch_size=32, box_width=2, box_height=2, sequence_length=50):
        '''
        For testing performance, returns a batch of sample trajectories

        Returns:
            positions: jax.numpy array of shape [sequence_length, batch_size, 2]
        '''
        positions = self.generate_trajectory(box_width, box_height, batch_size, sequence_length)
        positions = jnp.array(positions).transpose(1, 0, 2)
        return positions

    def generate_dataset(self, savepath, num_batches=100, batch_size=512, sequence_length=50, box_width=2, box_height=2):
        '''
        Generate a dataset of trajectory batches and save to disk.

        Args:
            savepath: Directory to save the dataset
            num_batches: Number of batches to generate
            batch_size: Size of each batch
            sequence_length: Length of each trajectory
            box_width: Width of the environment box
            box_height: Height of the environment box
        '''
        # Create save directory if it doesn't exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        print(f"Generating {num_batches} batches of size {batch_size}...")

        for batch_idx in range(num_batches):
            positions = self.generate_trajectory(box_width, box_height, batch_size, sequence_length)
            # Transpose to [sequence_length, batch_size, 2]
            positions = positions.transpose(1, 0, 2)

            # Save batch to disk as pickle (JAX arrays can be pickled)
            batch_path = os.path.join(savepath, f'batch_{batch_idx:05d}.pkl')
            with open(batch_path, 'wb') as f:
                pickle.dump(positions, f)

            if (batch_idx + 1) % 10 == 0:
                print(f"Generated {batch_idx + 1}/{num_batches} batches")

        # Save metadata
        metadata = {
            'num_batches': num_batches,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'box_width': box_width,
            'box_height': box_height,
        }
        with open(os.path.join(savepath, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f"Dataset generation complete! Saved to {savepath}")
        print(f"Total samples: {num_batches * batch_size}")
