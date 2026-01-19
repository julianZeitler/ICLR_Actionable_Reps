# -*- coding: utf-8 -*-
import jax.numpy as jnp
import numpy as np
import os
import json
import pickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class TrajectoryDataset:
    """
    Efficient dataset loader for pre-generated trajectory data with LRU caching
    and optional automatic prefetching with multi-worker support.
    """
    def __init__(self, dataset_path, cache_size=128, num_workers=0, prefetch_batches=0):
        """
        Args:
            dataset_path: Path to the directory containing the dataset
            cache_size: Number of batch files to keep in memory (default: 128)
            num_workers: Number of parallel workers for prefetching. 0 disables prefetching (default: 0)
            prefetch_batches: Number of batches to prefetch ahead when accessing data (default: 0)
        """
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches

        # Load metadata
        with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        self.num_batches = self.metadata['num_batches']
        self.batch_size = self.metadata['batch_size']
        self.sequence_length = self.metadata['sequence_length']

        # Create cached load function with specified cache size
        self._load_batch = lru_cache(maxsize=cache_size)(self._load_batch_uncached)

        # Initialize eager loading components
        self._executor = None
        self._shutdown = False
        self._last_accessed_batch = -1

        if self.num_workers > 0 and self.prefetch_batches > 0:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self._prefetch_futures = set()

    def __len__(self):
        return self.num_batches * self.batch_size

    def _load_batch_uncached(self, batch_idx):
        """Load a batch file from disk (uncached)"""
        batch_path = os.path.join(self.dataset_path, f'batch_{batch_idx:05d}.pkl')
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f)
        return batch

    def _trigger_prefetch(self, current_batch_idx):
        """Automatically prefetch upcoming batches based on access pattern"""
        if self._executor is None or self.prefetch_batches == 0:
            return

        # Clean up completed futures
        self._prefetch_futures = {f for f in self._prefetch_futures if not f.done()}

        # Determine which batches to prefetch
        prefetch_start = current_batch_idx + 1
        prefetch_end = min(prefetch_start + self.prefetch_batches, self.num_batches)

        # Submit prefetch jobs for upcoming batches
        for batch_idx in range(prefetch_start, prefetch_end):
            # Only prefetch if not already loading
            future = self._executor.submit(self._load_batch, batch_idx)
            self._prefetch_futures.add(future)

    def __getitem__(self, idx):
        # Determine which batch file and which sample within that batch
        batch_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size

        # Trigger prefetch for upcoming batches
        if batch_idx != self._last_accessed_batch:
            self._trigger_prefetch(batch_idx)
            self._last_accessed_batch = batch_idx

        # Load the batch (from cache if available)
        batch = self._load_batch(batch_idx)

        # Extract the specific trajectory (shape: [sequence_length, 2])
        trajectory = batch[:, sample_idx, :]

        return trajectory

    def get_batch(self, batch_idx):
        """Get an entire batch file"""
        # Trigger prefetch for upcoming batches
        if batch_idx != self._last_accessed_batch:
            self._trigger_prefetch(batch_idx)
            self._last_accessed_batch = batch_idx

        return self._load_batch(batch_idx)

    def clear_cache(self):
        """Clear the cache to free memory"""
        self._load_batch.cache_clear()

    def cache_info(self):
        """Get cache statistics"""
        return self._load_batch.cache_info()

    def shutdown(self):
        """Shutdown the prefetch thread pool and cleanup resources"""
        if self._executor is not None:
            self._shutdown = True
            self._executor.shutdown(wait=True)
            self._executor = None
            self._prefetch_futures = set()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.shutdown()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False


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

    def generate_raster_points(self, box_width, box_height, grid_resolution):
        '''
        Generate equally-spaced raster points in a grid pattern.

        Args:
            box_width: Width of the environment box
            box_height: Height of the environment box
            grid_resolution: Number of points along each dimension (e.g., 10 creates 10x10 grid)

        Returns:
            points: numpy array of shape [grid_resolution^2, 2]
                   Contains (x, y) coordinates for each grid point
        '''
        x = np.linspace(-box_width / 2, box_width / 2, grid_resolution)
        y = np.linspace(-box_height / 2, box_height / 2, grid_resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        return points

    def generate_raster_trajectories(self, box_width, box_height, grid_resolution, batch_size):
        '''
        Generate raster point trajectories (seq_len=1, equally spaced points).

        Args:
            box_width: Width of the environment box
            box_height: Height of the environment box
            grid_resolution: Number of points along each dimension
            batch_size: Number of points to sample (with replacement if > grid_resolution^2)

        Returns:
            positions: numpy array of shape [batch_size, 1, 2]
                      Each trajectory is a single point from the raster grid
        '''
        raster_points = self.generate_raster_points(box_width, box_height, grid_resolution)

        # Sample with replacement if batch_size > number of grid points
        num_points = raster_points.shape[0]
        indices = np.random.choice(num_points, size=batch_size, replace=(batch_size > num_points))

        # Reshape to [batch_size, 1, 2] for seq_len=1
        positions = raster_points[indices][:, np.newaxis, :]
        return positions

    def generate_snake_trajectories(self, box_width, box_height, grid_resolution, batch_size):
        '''
        Generate snake-pattern trajectories connecting raster points.

        Args:
            box_width: Width of the environment box
            box_height: Height of the environment box
            grid_resolution: Number of points along each dimension
            batch_size: Number of trajectories to generate

        Returns:
            positions: numpy array of shape [batch_size, grid_resolution^2, 2]
                      Each trajectory follows a snake pattern through all grid points
        '''
        # Generate base raster grid
        x = np.linspace(-box_width / 2, box_width / 2, grid_resolution)
        y = np.linspace(-box_height / 2, box_height / 2, grid_resolution)

        # Create snake pattern: alternate row directions
        snake_points = []
        for i, y_val in enumerate(y):
            if i % 2 == 0:
                # Left to right
                for x_val in x:
                    snake_points.append([x_val, y_val])
            else:
                # Right to left
                for x_val in reversed(x):
                    snake_points.append([x_val, y_val])

        snake_points = np.array(snake_points)  # Shape: [grid_resolution^2, 2]
        sequence_length = snake_points.shape[0]

        # Create batch_size trajectories with random starting points and rotations
        positions = np.zeros([batch_size, sequence_length, 2])

        for b in range(batch_size):
            # Random rotation
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            # Random starting offset (small)
            offset = np.random.uniform(-0.1, 0.1, size=2)

            # Apply rotation and offset
            rotated_points = snake_points @ rotation_matrix.T + offset
            positions[b] = rotated_points

        return positions

    def generate_spiral_trajectories(self, box_width, box_height, num_turns, points_per_turn, batch_size):
        '''
        Generate spiral-pattern trajectories.

        Args:
            box_width: Width of the environment box (defines max radius)
            box_height: Height of the environment box (defines max radius)
            num_turns: Number of complete spiral turns
            points_per_turn: Number of points per turn
            batch_size: Number of trajectories to generate

        Returns:
            positions: numpy array of shape [batch_size, num_turns * points_per_turn, 2]
                      Each trajectory follows a spiral pattern
        '''
        sequence_length = num_turns * points_per_turn
        max_radius = min(box_width, box_height) / 2 * 0.9  # Stay within bounds

        # Create base spiral
        angles = np.linspace(0, num_turns * 2 * np.pi, sequence_length)
        radii = np.linspace(0, max_radius, sequence_length)

        spiral_points = np.stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ], axis=-1)  # Shape: [sequence_length, 2]

        # Create batch_size trajectories with random rotations and directions
        positions = np.zeros([batch_size, sequence_length, 2])

        for b in range(batch_size):
            # Random rotation
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            # Random direction (inward or outward spiral)
            if np.random.rand() > 0.5:
                spiral_to_use = spiral_points[::-1]  # Reverse for inward spiral
            else:
                spiral_to_use = spiral_points

            # Apply rotation
            rotated_spiral = spiral_to_use @ rotation_matrix.T
            positions[b] = rotated_spiral

        return positions

    def generate_validation_dataset(self, savepath, mode='raster',
                                    box_sizes=None, grid_resolution=10,
                                    num_turns=5, points_per_turn=20,
                                    num_batches=10, batch_size=100):
        '''
        Generate validation datasets with interpretable trajectory patterns.
        Saves in TrajectoryDataset format for easy loading.

        Args:
            savepath: Directory to save the validation dataset
            mode: Type of trajectory pattern ('raster', 'snake', or 'spiral')
            box_sizes: List of (width, height) tuples for environment sizes
                      If None, creates datasets for single default size (2, 2)
            grid_resolution: Grid resolution for raster/snake modes
            num_turns: Number of spiral turns (for spiral mode)
            points_per_turn: Points per turn (for spiral mode)
            num_batches: Number of batches to generate
            batch_size: Number of samples per batch

        Returns:
            List of dataset paths created
        '''
        if box_sizes is None:
            box_sizes = [(2, 2)]  # Default size

        created_datasets = []

        for box_width, box_height in box_sizes:
            # Create subdirectory for this configuration
            if mode == 'raster':
                config_name = f"raster_grid{grid_resolution}_box{box_width}x{box_height}"
                sequence_length = 1
            elif mode == 'snake':
                config_name = f"snake_grid{grid_resolution}_box{box_width}x{box_height}"
                sequence_length = grid_resolution ** 2
            elif mode == 'spiral':
                config_name = f"spiral_turns{num_turns}_ppt{points_per_turn}_box{box_width}x{box_height}"
                sequence_length = num_turns * points_per_turn
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'raster', 'snake', or 'spiral'")

            dataset_path = os.path.join(savepath, config_name)
            os.makedirs(dataset_path, exist_ok=True)

            print(f"Generating {mode} validation data for box size ({box_width}, {box_height})...")
            print(f"  Sequence length: {sequence_length}")

            # Generate and save batches
            for batch_idx in range(num_batches):
                if mode == 'raster':
                    positions = self.generate_raster_trajectories(
                        box_width, box_height, grid_resolution, batch_size
                    )
                elif mode == 'snake':
                    positions = self.generate_snake_trajectories(
                        box_width, box_height, grid_resolution, batch_size
                    )
                elif mode == 'spiral':
                    positions = self.generate_spiral_trajectories(
                        box_width, box_height, num_turns, points_per_turn, batch_size
                    )

                # Save batch
                batch_path = os.path.join(dataset_path, f'batch_{batch_idx:05d}.pkl')
                with open(batch_path, 'wb') as f:
                    pickle.dump(positions, f)

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Generated {batch_idx + 1}/{num_batches} batches")

            # Save metadata in TrajectoryDataset format
            metadata = {
                'num_batches': num_batches,
                'batch_size': batch_size,
                'sequence_length': sequence_length,
                'box_width': box_width,
                'box_height': box_height,
                'mode': mode,
            }

            # Add mode-specific metadata
            if mode == 'raster' or mode == 'snake':
                metadata['grid_resolution'] = grid_resolution
            elif mode == 'spiral':
                metadata['num_turns'] = num_turns
                metadata['points_per_turn'] = points_per_turn

            with open(os.path.join(dataset_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            created_datasets.append(dataset_path)
            print(f"  Saved to {dataset_path}")

        print(f"\nValidation dataset generation complete!")
        print(f"Total configurations: {len(created_datasets)}")
        print(f"Total samples per config: {num_batches * batch_size}")

        return created_datasets

    def visualize_trajectories(self, positions, box_width=2, box_height=2,
                               num_trajectories=None, show_arrows=True,
                               show_start=True, show_end=True,
                               title=None, figsize=(10, 10), save_path=None):
        '''
        Visualize trajectory data.

        Args:
            positions: numpy array of shape [batch_size, sequence_length, 2]
            box_width: Width of environment box (for drawing boundaries)
            box_height: Height of environment box (for drawing boundaries)
            num_trajectories: Number of trajectories to plot (None = all, or specify max)
            show_arrows: Whether to show direction arrows
            show_start: Whether to mark starting points
            show_end: Whether to mark ending points
            title: Plot title
            figsize: Figure size tuple
            save_path: If provided, save figure to this path

        Returns:
            fig, ax: Matplotlib figure and axis objects
        '''
        fig, ax = plt.subplots(figsize=figsize)

        batch_size, sequence_length, _ = positions.shape

        # Determine how many trajectories to plot
        if num_trajectories is None:
            num_trajectories = min(batch_size, 50)  # Cap at 50 for visibility
        else:
            num_trajectories = min(num_trajectories, batch_size)

        # Color map for trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))

        for i in range(num_trajectories):
            traj = positions[i]  # Shape: [sequence_length, 2]

            # Plot trajectory
            ax.plot(traj[:, 0], traj[:, 1], '-', alpha=0.6, color=colors[i], linewidth=1.5)

            # Mark start point
            if show_start:
                ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], markersize=8,
                       markeredgecolor='black', markeredgewidth=1.5, label='Start' if i == 0 else '')

            # Mark end point
            if show_end and sequence_length > 1:
                ax.plot(traj[-1, 0], traj[-1, 1], 's', color=colors[i], markersize=8,
                       markeredgecolor='black', markeredgewidth=1.5, label='End' if i == 0 else '')

            # Add direction arrows
            if show_arrows and sequence_length > 1:
                # Show arrows at regular intervals
                arrow_interval = max(1, sequence_length // 5)
                for j in range(0, sequence_length - 1, arrow_interval):
                    dx = traj[j + 1, 0] - traj[j, 0]
                    dy = traj[j + 1, 1] - traj[j, 1]
                    ax.arrow(traj[j, 0], traj[j, 1], dx * 0.5, dy * 0.5,
                            head_width=0.1, head_length=0.1, fc=colors[i], ec=colors[i],
                            alpha=0.4, linewidth=0.5)

        # Draw box boundaries
        rect = Rectangle((-box_width / 2, -box_height / 2), box_width, box_height,
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
                        label='Environment boundary')
        ax.add_patch(rect)

        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.set_xlabel('X position (m)', fontsize=12)
        ax.set_ylabel('Y position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Set title
        if title is None:
            title = f'Trajectories (showing {num_trajectories}/{batch_size})'
        ax.set_title(title, fontsize=14)

        # Add legend
        if show_start or show_end or True:
            ax.legend(loc='upper right')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig, ax

    def visualize_trajectory_grid(self, positions, box_width=2, box_height=2,
                                  grid_size=(4, 4), title=None,
                                  figsize=(16, 16), save_path=None):
        '''
        Visualize multiple trajectories in a grid layout (one per subplot).

        Args:
            positions: numpy array of shape [batch_size, sequence_length, 2]
            box_width: Width of environment box
            box_height: Height of environment box
            grid_size: Tuple (rows, cols) for subplot grid
            title: Overall figure title
            figsize: Figure size tuple
            save_path: If provided, save figure to this path

        Returns:
            fig, axes: Matplotlib figure and axes objects
        '''
        batch_size, sequence_length, _ = positions.shape
        rows, cols = grid_size
        num_plots = min(rows * cols, batch_size)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows * cols > 1 else [axes]

        for idx in range(num_plots):
            ax = axes[idx]
            traj = positions[idx]  # Shape: [sequence_length, 2]

            # Plot trajectory with time-based color gradient
            if sequence_length > 1:
                points = traj.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                from matplotlib.collections import LineCollection
                lc = LineCollection(segments, cmap='viridis', linewidth=2)
                lc.set_array(np.linspace(0, 1, sequence_length - 1))
                ax.add_collection(lc)

            # Mark start and end
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10,
                   markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=10)
            if sequence_length > 1:
                ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10,
                       markeredgecolor='black', markeredgewidth=1.5, label='End', zorder=10)

            # Draw box boundaries
            rect = Rectangle((-box_width / 2, -box_height / 2), box_width, box_height,
                            linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)

            # Set limits with some padding
            padding = 0.2
            ax.set_xlim(-box_width / 2 - padding, box_width / 2 + padding)
            ax.set_ylim(-box_height / 2 - padding, box_height / 2 + padding)

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Trajectory {idx}', fontsize=10)

            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        # Hide unused subplots
        for idx in range(num_plots, rows * cols):
            axes[idx].axis('off')

        if title:
            fig.suptitle(title, fontsize=16, y=0.995)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig, axes

    def visualize_trajectory_heatmap(self, positions, box_width=2, box_height=2,
                                     bins=50, title=None, figsize=(10, 8),
                                     save_path=None):
        '''
        Create a heatmap showing the spatial distribution of trajectory positions.

        Args:
            positions: numpy array of shape [batch_size, sequence_length, 2]
            box_width: Width of environment box
            box_height: Height of environment box
            bins: Number of bins for 2D histogram
            title: Plot title
            figsize: Figure size tuple
            save_path: If provided, save figure to this path

        Returns:
            fig, ax: Matplotlib figure and axis objects
        '''
        fig, ax = plt.subplots(figsize=figsize)

        # Flatten all positions
        all_positions = positions.reshape(-1, 2)  # Shape: [batch_size * sequence_length, 2]

        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            all_positions[:, 0], all_positions[:, 1],
            bins=bins,
            range=[[-box_width / 2, box_width / 2], [-box_height / 2, box_height / 2]]
        )

        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(h.T, origin='lower', extent=extent, cmap='hot', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Occupancy count', fontsize=12)

        # Draw box boundaries
        rect = Rectangle((-box_width / 2, -box_height / 2), box_width, box_height,
                        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--',
                        label='Environment boundary')
        ax.add_patch(rect)

        ax.set_xlabel('X position (m)', fontsize=12)
        ax.set_ylabel('Y position (m)', fontsize=12)

        if title is None:
            title = f'Trajectory Occupancy Heatmap ({len(positions)} trajectories)'
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig, ax
