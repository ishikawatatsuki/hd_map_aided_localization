import geopandas as gpd
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors

@dataclass
class MatchingResult:
    success: bool
    transformation: Optional[np.ndarray] = None
    corrected_sources: Optional[np.ndarray] = None
    original_sources: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    inlier_count: Optional[int] = None
    inlier_ratio: Optional[float] = None
    mean_error: Optional[float] = None
    rms_error_before: Optional[float] = None
    rms_error_after: Optional[float] = None
    improvement: Optional[float] = None
    metadata: Optional[dict] = None

class WindowedICPMapMatcher:
    """
    Windowed ICP-based map matcher for dead-reckoning correction.
    
    Accumulates N point correspondences (estimated positions and their OSM matches),
    then applies RANSAC-ICP to compute a transformation matrix that corrects
    the accumulated estimates.
    
    Features:
    - Circular buffer for point correspondences
    - Configurable window size and matching frequency
    - RANSAC outlier rejection for robustness
    - Statistics tracking (inlier ratio, RMS error, etc.)
    - Retrieval of corrected estimates
    
    Usage:
        matcher = WindowedICPMapMatcher(window_size=50, match_frequency=50)
        
        # In estimation loop:
        for pose in trajectory:
            osm_match = project_to_nearest_edge(pose, edges)
            if osm_match:
                matcher.add_correspondence(
                    source=[pose.lon, pose.lat],
                    target=[osm_match.lon, osm_match.lat],
                    metadata={'timestamp': pose.time, 'idx': i}
                )
                
                if matcher.should_match():
                    result = matcher.compute_and_apply_icp()
                    if result['success']:
                        corrected = result['corrected_sources']
                        T = result['transformation']
    """
    
    def __init__(self, 
                 window_size=50,
                 match_frequency=None,
                 use_ransac=True,
                 ransac_iterations=150,
                 ransac_sample_size=3,
                 inlier_threshold=0.00015,
                 min_inlier_ratio=0.5,
                 icp_max_iterations=20,
                 icp_tolerance=0.0001,
                 get_3d_transformation_matrix=False,
                 verbose=False,
                 max_consecutive_misses=5,
                 min_point_spacing=1.0,
                 max_source_jump=20.0,
                 rise_flag_by_travel_distance=False,
                 distance_threshold=0.01,
                 time_threshold=10.0
                ):
        """
        Parameters
        ----------
        window_size : int
            Number of correspondences to accumulate before matching
        match_frequency : int, optional
            Perform matching every N correspondences. If None, matches when window is full.
        use_ransac : bool
            Whether to use RANSAC outlier rejection (True) or standard ICP (False)
        ransac_iterations : int
            Number of RANSAC iterations
        ransac_sample_size : int
            Minimum points to sample per RANSAC iteration
        inlier_threshold : float
            Distance threshold for inlier classification (in coordinate units)
        min_inlier_ratio : float
            Minimum inlier ratio [0-1] required to accept transformation
        icp_max_iterations : int
            Maximum ICP iterations
        icp_tolerance : float
            ICP convergence tolerance
        max_consecutive_misses : int
            Maximum consecutive matches that fail before resetting buffer
        min_point_spacing : float, optional
            Minimum spacing between points to consider for matching (in coordinate units). This can handle static points or very close points that might cause issues.
        max_source_jump : float, optional
            Maximum allowed jump in source points between matches (in coordinate units). This can help detect when the source points have diverged too much, indicating a potential reset or loss of tracking.
        rise_flag_by_travel_distance : bool, optional
            If True, trigger matching when distance threshold is reached in addition to correspondence count.
        distance_threshold : float, optional
            Total distance traveled (in coordinate units) to trigger matching when rise_flag_by_travel_distance is True.
        time_threshold : float, optional
            Maximum time gap (in seconds) allowed between correspondences. Older correspondences are cleared if gap exceeds this.
        """
        self.window_size = window_size
        self.match_frequency = match_frequency if match_frequency else window_size
        self.use_ransac = use_ransac
        self.ransac_iterations = ransac_iterations
        self.ransac_sample_size = ransac_sample_size
        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.icp_max_iterations = icp_max_iterations
        self.icp_tolerance = icp_tolerance
        self.get_3d_transformation_matrix = get_3d_transformation_matrix
        self.max_consecutive_misses = max_consecutive_misses
        self.min_point_spacing = min_point_spacing
        self.max_source_jump = max_source_jump
        self.rise_flag_by_travel_distance = rise_flag_by_travel_distance
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold
        
        # Circular buffers for correspondences
        self.sources = []  # Estimated positions (dead reckoning)
        self.targets = []  # OSM matched positions
        self.metadata = []  # Optional metadata per correspondence
        
        # Counters
        self.total_matches = 0
        self.added_since_match = 0
        self.consecutive_misses = 0
        self.cumulative_distance = 0.0  # Accumulated travel distance since last match
        
        # Statistics
        self.stats_history = []
        
        # Last result
        self.last_result = None

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        if self.verbose:
            msg = f"Initialized WindowedICPMapMatcher with window_size={window_size}, match_frequency={self.match_frequency}, use_ransac={use_ransac}"
            if self.rise_flag_by_travel_distance:
                msg += f", distance_trigger={distance_threshold}m, time_threshold={time_threshold}s"
            print(msg)


    def register_miss(self):
        """Call when no OSM correspondence is found for current frame."""
        self.consecutive_misses += 1
        if self.consecutive_misses >= self.max_consecutive_misses:
            if self.verbose:
                print(
                    f"Clearing ICP buffer after {self.consecutive_misses} consecutive misses"
                )
            self.clear_buffer(reset_added_since_match=True)
        
    def add_correspondence(self, source, target, metadata=None):
        """
        Add a correspondence pair to the buffer.
        
        Parameters
        ----------
        source : array-like, shape (2,) or (3,)
            Estimated position [lon, lat] or [x, y, z]
        target : array-like, shape (2,) or (3,)
            Matched OSM position
        metadata : dict, optional
            Additional information (timestamp, index, etc.).
            Expected to contain 'timestamp' (float, epoch seconds) if time-based clearing is enabled.
        """
        source = np.asarray(source)
        target = np.asarray(target)

        # Get current timestamp from metadata (if available)
        current_timestamp = None
        if metadata is not None and 'timestamp' in metadata:
            current_timestamp = metadata['timestamp']
        
        # Time-based buffer clearing: if gap between first and current correspondence exceeds threshold
        if self.time_threshold is not None and len(self.metadata) > 0 and current_timestamp is not None:
            first_timestamp = self.metadata[0].get('timestamp', None)
            if first_timestamp is not None:
                time_gap = current_timestamp - first_timestamp
                if time_gap > self.time_threshold:
                    if self.verbose:
                        print(f"Clearing ICP buffer: time gap {time_gap:.2f}s exceeds threshold {self.time_threshold}s")
                    self.clear_buffer(reset_added_since_match=True)
                    self.cumulative_distance = 0.0

        # Optional distance-based decimation
        if self.min_point_spacing is not None and len(self.sources) > 0:
            if np.linalg.norm(source - self.sources[-1]) < self.min_point_spacing:
                return  # ignore near-duplicate point

        # Optional jump guard
        if self.max_source_jump is not None and len(self.sources) > 0:
            if np.linalg.norm(source - self.sources[-1]) > self.max_source_jump:
                if self.verbose:
                    print("Large source jump detected; clearing ICP buffer")
                self.clear_buffer(reset_added_since_match=True)
                self.cumulative_distance = 0.0

        # valid correspondence resets miss streak
        self.consecutive_misses = 0

        # Accumulate travel distance
        if len(self.sources) > 0:
            distance_step = np.linalg.norm(source - self.sources[-1])
            self.cumulative_distance += distance_step

        # Maintain circular buffer
        if len(self.sources) >= self.window_size:
            self.sources.pop(0)
            self.targets.pop(0)
            self.metadata.pop(0)
        
        self.sources.append(source)
        self.targets.append(target)
        self.metadata.append(metadata if metadata else {})
        self.added_since_match += 1
    
    def should_match(self):
        """
        Check if we should perform ICP matching.
        
        Triggers matching when EITHER:
        - Accumulated correspondences reach match_frequency, OR
        - (if rise_flag_by_travel_distance enabled) Cumulative distance exceeds distance_threshold
        
        Returns
        -------
        bool
            True if matching should be performed
        """
        min_pts = max(3, self.ransac_sample_size)
        if len(self.sources) < min_pts:
            return False
        
        # Trigger by correspondence count
        count_trigger = self.added_since_match >= self.match_frequency
        
        # Trigger by travel distance
        distance_trigger = False
        if self.rise_flag_by_travel_distance:
            distance_trigger = self.cumulative_distance >= self.distance_threshold
        
        # Return True if either trigger is active
        return count_trigger or distance_trigger
    
    def get_buffer_size(self):
        """Get current number of correspondences in buffer."""
        return len(self.sources)
    
    def get_cumulative_distance(self):
        """Get cumulative travel distance accumulated since last match."""
        return self.cumulative_distance
    
    def get_buffer_info(self):
        """
        Get detailed information about current buffer state.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'buffer_size': int, number of correspondences
            - 'added_since_match': int, correspondences added since last match
            - 'cumulative_distance': float, total distance accumulated
            - 'distance_to_trigger': float, remaining distance to travel for distance-based trigger
            - 'correspondences_to_trigger': int, remaining correspondences for count-based trigger
        """
        distance_to_trigger = max(0, self.distance_threshold - self.cumulative_distance) if self.rise_flag_by_travel_distance else None
        correspondences_to_trigger = max(0, self.match_frequency - self.added_since_match)
        
        return {
            'buffer_size': len(self.sources),
            'added_since_match': self.added_since_match,
            'cumulative_distance': self.cumulative_distance,
            'distance_to_trigger': distance_to_trigger,
            'correspondences_to_trigger': correspondences_to_trigger,
        }
    
    def clear_buffer(self, reset_added_since_match=False):
        """Clear the correspondence buffer."""
        self.sources = []
        self.targets = []
        self.metadata = []
        self.consecutive_misses = 0
        if reset_added_since_match:
            self.added_since_match = 0
            self.cumulative_distance = 0.0

    def _transform_2d_to_3d(self, T_2d: np.ndarray) -> np.ndarray:
        """
        Convert a 2D homogeneous transformation (3x3) to a 3D
        homogeneous transformation (4x4).
        Returns:
        T_3d : np.ndarray, shape (4, 4)
            3D homogeneous rigid body transformation.
        """
        assert T_2d.shape == (3, 3), f"Expected (3,3), got {T_2d.shape}"

        R_2d = T_2d[:2, :2]  # 2x2 rotation
        t_2d = T_2d[:2, 2]   # 2x1 translation

        T_3d = np.eye(4)
        T_3d[0:2, 0:2] = R_2d  # xy rotation
        T_3d[0:2, 3] = t_2d    # xy translation
        # T_3d[2, 2] = 1.0     # z unchanged (already set by np.eye)
        # T_3d[3, 3] = 1.0     # homogeneous (already set by np.eye)

        return T_3d

    def compute_inliers(self, A, B, T, inlier_threshold):
        """
        Compute inlier mask based on distance after transformation.
        
        Parameters
        ----------
        A : np.ndarray, shape (N, m)
            Source points
        B : np.ndarray, shape (N, m)
            Target points (matched correspondences)
        T : np.ndarray, shape (m+1, m+1)
            Homogeneous transformation matrix
        inlier_threshold : float
            Maximum distance for a point to be considered an inlier
            
        Returns
        -------
        inlier_mask : np.ndarray, shape (N,)
            Boolean mask where True indicates inlier
        inlier_distances : np.ndarray, shape (N,)
            Distance of each correspondence after transformation
        """
        assert A.shape == B.shape
        m = A.shape[1]
        
        # Transform A
        A_h = np.hstack([A, np.ones((A.shape[0], 1))]).T  # (m+1, N)
        A_transformed = (T @ A_h)[:m, :].T  # (N, m)
        
        # Compute distances
        distances = np.linalg.norm(A_transformed - B, axis=1)
        
        # Inlier mask
        inlier_mask = distances < inlier_threshold
        
        return inlier_mask, distances
    
    def best_fit_transform(self, A, B):
        """
        Calculates the best-fit transform that maps points A onto points B.
        Input:
            A: Nxm numpy array of source points
            B: Nxm numpy array of destination points
        Output:
            T: (m+1)x(m+1) homogeneous transformation matrix
        """
        assert A.shape == B.shape
        m = A.shape[1]
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_B.reshape(-1,1) - np.dot(R, centroid_A.reshape(-1,1))
        T = np.eye(m+1)
        T[:m, :m] = R
        T[:m, -1] = t.ravel()
        return T


    def ransac_icp(self, A, B, max_iterations=20, tolerance=0.001, 
                ransac_iterations=100, ransac_sample_size=3, 
                inlier_threshold=0.01, min_inlier_ratio=0.5):
        """
        RANSAC-based robust ICP with outlier rejection.
        
        Parameters
        ----------
        A : np.ndarray, shape (N, m)
            Source points
        B : np.ndarray, shape (N, m) 
            Target points (matched correspondences)
        max_iterations : int
            Maximum ICP iterations
        tolerance : float
            ICP convergence tolerance
        ransac_iterations : int
            Number of RANSAC iterations
        ransac_sample_size : int
            Number of correspondences to sample per RANSAC iteration
        inlier_threshold : float
            Distance threshold for inlier classification (in same units as points)
        min_inlier_ratio : float
            Minimum ratio of inliers required to accept solution
            
        Returns
        -------
        T_final : np.ndarray, shape (m+1, m+1)
            Best homogeneous transformation
        inlier_mask : np.ndarray, shape (N,)
            Boolean mask of final inliers
        inlier_count : int
            Number of inliers in final solution
        """
        assert A.shape == B.shape
        assert A.shape[0] >= ransac_sample_size, "Not enough points for RANSAC"
        
        N = A.shape[0]
        m = A.shape[1]
        best_inlier_count = 0
        best_T = np.eye(m + 1)
        best_inlier_mask = np.zeros(N, dtype=bool)
        
        self.logger.debug(f"Running RANSAC-ICP with {ransac_iterations} RANSAC iterations...")
        
        # RANSAC loop
        for iter_idx in range(ransac_iterations):
            # Randomly sample correspondences
            sample_indices = np.random.choice(N, ransac_sample_size, replace=False)
            A_sample = A[sample_indices]
            B_sample = B[sample_indices]
            
            # Compute transformation from sample
            try:
                T_candidate = self.best_fit_transform(A_sample, B_sample)
            except np.linalg.LinAlgError:
                continue  # Skip if singular
                
            # Count inliers for this transformation
            inlier_mask, distances = self.compute_inliers(A, B, T_candidate, inlier_threshold)
            inlier_count = np.sum(inlier_mask)
            
            # Update best if we found more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_T = T_candidate
                best_inlier_mask = inlier_mask
        
        inlier_ratio = best_inlier_count / N
        self.logger.debug(f"RANSAC found {best_inlier_count}/{N} inliers ({inlier_ratio:.1%})")
        
        # Check if we have enough inliers
        if inlier_ratio < min_inlier_ratio:
            self.logger.warning(f"Inlier ratio {inlier_ratio:.1%} below threshold {min_inlier_ratio:.1%}")
            self.logger.debug("Returning identity transformation")
            return np.eye(m + 1), best_inlier_mask, best_inlier_count
        
        # Refine using ICP with inliers only
        self.logger.debug(f"Refining with ICP using {best_inlier_count} inliers...")
        A_inliers = A[best_inlier_mask]
        B_inliers = B[best_inlier_mask]
        
        # Run standard ICP on inliers (without visualization for speed)
        T_refined = self.icp_core(A_inliers, B_inliers, max_iterations, tolerance)
        
        # Recompute final inliers with refined transformation
        final_inlier_mask, final_distances = self.compute_inliers(A, B, T_refined, inlier_threshold)
        final_inlier_count = np.sum(final_inlier_mask)
        
        self.logger.debug(f"After refinement: {final_inlier_count}/{N} inliers ({final_inlier_count/N:.1%})")
        self.logger.debug(f"Mean inlier distance: {np.mean(final_distances[final_inlier_mask]):.6f}")
        
        return T_refined, final_inlier_mask, final_inlier_count

    def nearest_neighbor(self, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''
        # Ensure shapes are compatible for KNN, although they don't strictly need to be identical N
        assert src.shape[1] == dst.shape[1]
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def icp_core(self, A, B, max_iterations=20, tolerance=0.001):
        """
        Core ICP algorithm without visualization (faster for RANSAC refinement).
        
        Parameters
        ----------
        A : np.ndarray, shape (N, m)
            Source points
        B : np.ndarray, shape (N, m)
            Target points
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
            
        Returns
        -------
        T_final : np.ndarray, shape (m+1, m+1)
            Homogeneous transformation matrix
        """
        assert A.shape[1] == B.shape[1]
        m = A.shape[1]
        
        # Make points homogeneous
        src_h = np.ones((m+1, A.shape[0]))
        src_h[:m, :] = np.copy(A.T)
        
        dst = np.copy(B)
        prev_error = float('inf')
        
        for i in range(max_iterations):
            current_src = src_h[:m, :].T
            
            # Find nearest neighbors
            distances, indices = self.nearest_neighbor(current_src, dst)
            
            # Compute transformation
            T_step = self.best_fit_transform(current_src, dst[indices, :])
            
            # Update source points
            src_h = np.dot(T_step, src_h)
            
            # Check convergence
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
        
        # Final transformation
        T_final = self.best_fit_transform(A, src_h[:m, :].T)
        
        return T_final

    def compute_and_apply_icp(self):
        """
        Compute ICP transformation and apply to accumulated sources.
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'success': bool, whether ICP succeeded
            - 'transformation': np.ndarray (m+1, m+1), homogeneous transformation matrix
            - 'corrected_sources': np.ndarray (N, m), transformed source points
            - 'inlier_mask': np.ndarray (N,), boolean mask of inliers
            - 'inlier_count': int, number of inliers
            - 'inlier_ratio': float, ratio of inliers
            - 'mean_error': float, mean distance error for inliers
            - 'rms_error_before': float, RMS error before correction
            - 'rms_error_after': float, RMS error after correction
            - 'metadata': list of metadata for buffer points
        """
        if len(self.sources) < max(3, self.ransac_sample_size):
            self.logger.warning(f"Not enough correspondences to compute ICP (have {len(self.sources)}, need at least {max(3, self.ransac_sample_size)})")
            return  MatchingResult(success=False)
        
        self.added_since_match = 0
        self.cumulative_distance = 0.0  # Reset distance accumulator after match
        
        # Convert to numpy arrays
        A = np.array(self.sources)  # Source points (estimated)
        B = np.array(self.targets)  # Target points (OSM)
        
        # Compute RMS before correction
        rms_before = np.sqrt(np.mean(np.sum((A - B)**2, axis=1)))
        
        # Run ICP
        try:
            if self.use_ransac:
                T, inlier_mask, inlier_count = self.ransac_icp(
                    A, B,
                    max_iterations=self.icp_max_iterations,
                    tolerance=self.icp_tolerance,
                    ransac_iterations=self.ransac_iterations,
                    ransac_sample_size=self.ransac_sample_size,
                    inlier_threshold=self.inlier_threshold,
                    min_inlier_ratio=self.min_inlier_ratio
                )
            else:
                T = self.icp_core(A, B, self.icp_max_iterations, self.icp_tolerance)
                inlier_mask, _ = self.compute_inliers(A, B, T, self.inlier_threshold)
                inlier_count = np.sum(inlier_mask)
            
        except Exception as e:
            self.logger.warning(f"ICP failed: {e}")
            return MatchingResult(success=False)
        
        # Check if transformation is valid
        if np.allclose(T, np.eye(T.shape[0])):
            self.logger.warning("Warning: Identity transformation returned")
            return MatchingResult(success=False)
        
        # Apply transformation to sources
        m = A.shape[1]
        A_h = np.hstack([A, np.ones((A.shape[0], 1))]).T  # (m+1, N)
        A_corrected = (T @ A_h)[:m, :].T  # (N, m)
        
        # Compute RMS after correction
        rms_after = np.sqrt(np.mean(np.sum((A_corrected - B)**2, axis=1)))
        
        # Compute inlier error
        inlier_distances = np.linalg.norm(A_corrected - B, axis=1)
        mean_inlier_error = np.mean(inlier_distances[inlier_mask]) if np.any(inlier_mask) else np.inf
        
        inlier_ratio = inlier_count / len(A)
        
        # Store statistics
        stats = {
            'match_id': self.total_matches,
            'buffer_size': len(A),
            'inlier_count': inlier_count,
            'inlier_ratio': inlier_ratio,
            'rms_before': rms_before,
            'rms_after': rms_after,
            'mean_inlier_error': mean_inlier_error,
            'improvement': (rms_before - rms_after) / rms_before if rms_before > 0 else 0
        }
        self.stats_history.append(stats)
        self.total_matches += 1
        
        self.logger.debug(f"\n=== ICP Match #{self.total_matches} ===")
        
        # Log which trigger caused the match
        trigger_msg = "Trigger: "
        if self.added_since_match >= self.match_frequency and (not self.rise_flag_by_travel_distance or self.cumulative_distance < self.distance_threshold):
            trigger_msg += f"Correspondence count ({self.added_since_match} >= {self.match_frequency})"
        elif self.rise_flag_by_travel_distance and self.cumulative_distance >= self.distance_threshold:
            trigger_msg += f"Travel distance ({self.cumulative_distance:.4f} >= {self.distance_threshold})"
        else:
            trigger_msg += "Correspondence count & Travel distance"
        self.logger.debug(trigger_msg)
        
        self.logger.debug(f"Buffer size: {len(A)}")
        self.logger.debug(f"Inliers: {inlier_count}/{len(A)} ({inlier_ratio:.1%})")
        self.logger.debug(f"RMS before: {rms_before:.6f}")
        self.logger.debug(f"RMS after:  {rms_after:.6f}")
        self.logger.debug(f"Improvement: {stats['improvement']:.1%}")
        self.logger.debug(f"Mean inlier error: {mean_inlier_error:.6f}")
        
        if self.get_3d_transformation_matrix:
            T = self._transform_2d_to_3d(T)
            
        # Prepare result
        result = MatchingResult(
            success=True,
            transformation=T,
            corrected_sources=A_corrected,
            original_sources=A,
            targets=B,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_error=mean_inlier_error,
            rms_error_before=rms_before,
            rms_error_after=rms_after,
            improvement=stats['improvement'],
            metadata=self.metadata.copy()
        )

        self.last_result = result
        return result
        
    
    def get_statistics(self):
        """Get statistics history."""
        return self.stats_history.copy()
    
    def get_last_result(self):
        """Get the last ICP result."""
        return self.last_result
    
    def reset(self):
        """Reset the matcher (clear buffer and statistics)."""
        self.clear_buffer(reset_added_since_match=True)
        self.total_matches = 0
        self.cumulative_distance = 0.0
        self.stats_history = []
        self.last_result = None


if __name__ == "__main__":
    import os

    root_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion_ws"
    edges_map = gpd.read_parquet(f'{root_dir}/data/KITTI/raw/karlsruhe_edges.parquet')
    nodes_map = gpd.read_parquet(f'{root_dir}/data/KITTI/raw/karlsruhe_nodes.parquet')

    print(f"Loaded {len(edges_map)} edges and {len(nodes_map)} nodes")
    print(f"Total memory: {(edges_map.memory_usage(deep=True).sum() + nodes_map.memory_usage(deep=True).sum()) / 1024**2:.2f} MB")

    notebook_dir = "./notebooks"


    sample_estimated_points = np.load(os.path.join(notebook_dir, ".data/dead_reckoning_points.npy"))
    sample_osm_correction_points = np.load(os.path.join(notebook_dir, ".data/osm_correction_points.npy"))

    lon, lat = sample_estimated_points[0]
    lon_min, lon_max = lon - 0.001, lon + 0.001
    lat_min, lat_max = lat - 0.0001, lat + 0.0001

    edges_map_local = edges_map.cx[lon_min:lon_max, lat_min:lat_max].iloc[:3]
    print(f"Shape of sample estimated points: {sample_estimated_points.shape}")
    print(f"Shape of sample OSM correction points: {sample_osm_correction_points.shape}")

    plt.figure(figsize=(6, 6))
    edges_map_local.plot(color='lightgray', linewidth=1, label='Streets')
    plt.scatter(sample_estimated_points[:, 0], sample_estimated_points[:, 1], label="Estimated Points", color="red", s=1)
    plt.scatter(sample_osm_correction_points[:, 0], sample_osm_correction_points[:, 1], label="OSM Correction Points", color="blue", s=1)
    plt.legend()
    plt.title("Sample Correspondences")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()
    plt.savefig(os.path.join(notebook_dir, ".data/sample_correspondences.png"), dpi=300)