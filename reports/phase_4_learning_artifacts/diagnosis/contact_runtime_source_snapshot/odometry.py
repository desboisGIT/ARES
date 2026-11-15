"""CPU depth ICP with an optional body-gyro rotation prior.

This is a local depth odometry baseline, not full stereo visual-inertial SLAM.
Accelerometer translation, bias estimation and loop closure are unsupported.
"""

import math
import numpy as np

from .recording import CameraIntrinsics


# Optical camera X right/Y down/Z forward -> body X forward/Y left/Z up.
CAMERA_TO_BODY = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def _rotation(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-14:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _rotation_vector(rotation):
    cosine = np.clip((np.trace(rotation) - 1.) / 2., -1., 1.)
    angle = math.acos(float(cosine))
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-7:
        return skew / 2.
    return skew * angle / (2. * math.sin(angle))


def validate_pose(pose):
    value = np.asarray(pose, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("pose must be a finite 4x4 camera-to-world transform")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("pose must have a homogeneous final row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("pose rotation must be right-handed and orthonormal")
    return value.copy()


def depth_points(depth_m, camera):
    depth = np.asarray(depth_m)
    if depth.shape != (camera.height, camera.width) or depth.dtype.kind not in "fi":
        raise ValueError("depth shape must match camera; expected numeric optical Z")
    if np.isinf(depth).any() or np.any(np.isfinite(depth) & (depth <= 0)):
        raise ValueError("depth must contain positive optical Z or NaN")
    y, x = np.indices(depth.shape)
    return np.stack(((x - camera.cx) * depth / camera.fx,
                     (y - camera.cy) * depth / camera.fy, depth), axis=-1)


def _cloud(depth, camera):
    points = depth_points(depth, camera)
    if camera.width < 3 or camera.height < 3:
        return points, np.full(points.shape, np.nan)
    # A plane has affine inverse optical depth. Fit normals locally without
    # moving measured points: smoothing points across plane intersections biases
    # registration even when the resulting residual appears deceptively small.
    radius = 2
    offsets_y, offsets_x = np.indices((5, 5)) - radius
    design = np.stack((np.ones(25), offsets_x.ravel(), offsets_y.ravel()), axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(depth.astype(float), radius,
                                                               constant_values=np.nan), (5, 5))
    windows = windows.reshape(camera.height, camera.width, 25)
    valid = np.isfinite(windows) & np.isfinite(depth[..., None])
    valid &= np.abs(windows - depth[..., None]) < np.maximum(.15, .15 * depth[..., None])
    weights = valid * np.exp(-(offsets_x.ravel() ** 2 + offsets_y.ravel() ** 2) / 4.)
    inverse = np.divide(1., windows, out=np.zeros_like(windows), where=valid)
    matrix = np.einsum("...k,ki,kj->...ij", weights, design, design)
    rhs = np.einsum("...k,...k,ki->...i", weights, inverse, design)
    coefficients = np.linalg.solve(matrix + np.eye(3) * 1e-10, rhs[..., None])[..., 0]
    a, b, c = np.moveaxis(coefficients, -1, 0)
    y, x = np.indices(depth.shape)
    normals = np.stack((b * camera.fx, c * camera.fy,
                        a - b * (x - camera.cx) - c * (y - camera.cy)), axis=-1)
    length = np.linalg.norm(normals, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals /= length[..., None]
    fit = np.einsum("...i,ki->...k", coefficients, design)
    error = np.sqrt(np.sum(weights * (inverse - fit) ** 2, axis=-1) /
                    np.maximum(np.sum(weights, axis=-1), 1.)) * depth ** 2
    accepted = (valid.sum(axis=-1) >= 6) & (a > 0) & (error < np.maximum(.02, .006 * depth))
    normals[~accepted] = np.nan
    points[~accepted] = np.nan
    return points, normals


class DepthInertialOdometry:
    """Frame-to-keyframe projective point-to-plane ICP, NumPy CPU only.

    A failed frame never changes the accepted pose/reference. The returned pose
    on failure is stale and can_integrate=False. No GT or map is used.
    """

    def __init__(self, camera, *, initial_pose=None, max_points=900, max_iterations=18,
                 max_correspondence_m=.4, max_rmse_m=.10, max_condition=1e7,
                 gyro_noise_std_rps=.005, gyro_min_angle_std_rad=.0005,
                 keyframe_translation_m=.1, keyframe_rotation_rad=.1,
                 min_normal_span_ratio=.01):
        if not isinstance(camera, CameraIntrinsics):
            raise ValueError("camera must be CameraIntrinsics")
        if type(max_points) is not int or max_points < 30 or type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("require max_points >=30 and max_iterations >=1")
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in
               (max_correspondence_m, max_rmse_m, max_condition, gyro_noise_std_rps,
                gyro_min_angle_std_rad, keyframe_translation_m, keyframe_rotation_rad)):
            raise ValueError("ICP thresholds must be finite and positive")
        if isinstance(min_normal_span_ratio, bool) or not math.isfinite(min_normal_span_ratio) or not 0 < min_normal_span_ratio < 1:
            raise ValueError("min_normal_span_ratio must be finite and between zero and one")
        self.camera = camera
        self.pose = validate_pose(np.eye(4) if initial_pose is None else initial_pose)
        self.max_points, self.max_iterations = max_points, max_iterations
        self.max_correspondence_m, self.max_rmse_m, self.max_condition = max_correspondence_m, max_rmse_m, max_condition
        self._reference = None
        self._last_timestamp = None
        self._reference_timestamp = None
        self._reference_pose = self.pose.copy()
        self._last_gyro = None
        self._gyro_rotation = np.eye(3)
        self._gyro_chain_valid = True
        self.gyro_noise_std_rps, self.gyro_min_angle_std_rad = gyro_noise_std_rps, gyro_min_angle_std_rad
        self.keyframe_translation_m, self.keyframe_rotation_rad = keyframe_translation_m, keyframe_rotation_rad
        self.min_normal_span_ratio = min_normal_span_ratio
        self._normal_span_ratio, self._normal_span_rank = None, 0

    def _sample(self, points, normals):
        valid = np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1)
        p, n = points[valid], normals[valid]
        if len(p) > self.max_points:
            indices = np.linspace(0, len(p) - 1, self.max_points, dtype=int)
            p, n = p[indices], n[indices]
        return p, n

    def _metrics(self, points, normals, residual):
        jacobian = np.concatenate((np.cross(points, normals), normals), axis=1)
        information = jacobian.T @ jacobian
        eigenvalues = np.linalg.eigvalsh(information)
        rank = int(np.sum(eigenvalues > max(float(eigenvalues[-1]), 1.) * 1e-7))
        normal_eigenvalues = np.linalg.eigvalsh(normals.T @ normals)
        self._normal_span_ratio = float(normal_eigenvalues[0] / normal_eigenvalues[-1])
        self._normal_span_rank = int(np.sum(normal_eigenvalues > normal_eigenvalues[-1] * self.min_normal_span_ratio))
        # Numerical rank alone treats independent depth noise or mixed corner
        # normals as a real lateral constraint. Require three supported normal
        # directions before claiming all three translation axes observable.
        rank = min(rank, 3 + self._normal_span_rank)
        condition = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 1e-12 else None
        covariance = None
        if rank == 6:
            variance = max(float(np.mean(residual ** 2)), 1e-8)
            covariance = np.linalg.inv(information) * variance
        return jacobian, rank, condition, covariance

    def _result(self, status, reason, count=0, rmse=None, rank=0, condition=None, covariance=None, iterations=0):
        return {"pose": self.pose.copy(), "status": status, "reason": reason,
                "can_integrate": status in ("initialized", "tracking"), "rmse_m": rmse,
                "correspondences": int(count), "rank": rank, "condition_number": condition,
                "covariance": covariance, "iterations": iterations,
                "normal_span_ratio": self._normal_span_ratio, "normal_span_rank": self._normal_span_rank,
                "accelerometer_translation_used": False, "gyro_prior_used": self._gyro_used}

    def update(self, depth_m, timestamp_s, imu_accel_mps2=None, imu_gyro_rps=None):
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float, np.number)) or not math.isfinite(timestamp_s):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp_s <= self._last_timestamp:
            raise ValueError("timestamps must strictly increase")
        for value in (imu_accel_mps2, imu_gyro_rps):
            if value is not None and (np.asarray(value).shape != (3,) or not np.isfinite(value).all()):
                raise ValueError("IMU sample must be a finite body-frame vector of length 3")
        points, normals = _cloud(np.asarray(depth_m), self.camera)
        self._normal_span_ratio, self._normal_span_rank = None, 0
        gyro = None if imu_gyro_rps is None else np.asarray(imu_gyro_rps, dtype=float)
        if self._last_timestamp is not None and self._reference is not None:
            if gyro is not None:
                dt_update = float(timestamp_s) - self._last_timestamp
                angular_velocity = gyro if self._last_gyro is None else (gyro + self._last_gyro) / 2.
                increment = CAMERA_TO_BODY.T @ _rotation(angular_velocity * dt_update) @ CAMERA_TO_BODY
                self._gyro_rotation = self._gyro_rotation @ increment
            else:
                self._gyro_chain_valid = False
        self._last_gyro = None if gyro is None else gyro.copy()
        self._last_timestamp = float(timestamp_s)
        self._gyro_used = gyro is not None and self._reference is not None and self._gyro_chain_valid
        p, n = self._sample(points, normals)
        if len(p) < 30:
            return self._result("lost", "insufficient_depth", len(p))
        if self._reference is None:
            _, rank, condition, covariance = self._metrics(p, n, np.zeros(len(p)))
            if rank < 6 or condition is None or condition > self.max_condition:
                return self._result("lost", "unobservable_geometry", len(p), rank=rank, condition=condition)
            self._reference = (points, normals)
            self._reference_timestamp = float(timestamp_s)
            self._reference_pose = self.pose.copy()
            self._gyro_rotation, self._gyro_chain_valid = np.eye(3), True
            return self._result("initialized", "first_observable_frame", len(p), rank=rank,
                                condition=condition, covariance=covariance)
        transform = np.linalg.inv(self._reference_pose) @ self.pose
        if self._gyro_used:
            dt = float(timestamp_s) - self._reference_timestamp
            transform[:3, :3] = self._gyro_rotation
        rotation_prior = transform[:3, :3].copy()
        reference_points, reference_normals = self._reference

        def correspondences():
            transformed = p @ transform[:3, :3].T + transform[:3, 3]
            in_front = transformed[:, 2] > 1e-5
            safe_z = np.where(in_front, transformed[:, 2], 1.)
            u = np.rint(self.camera.fx * transformed[:, 0] / safe_z + self.camera.cx).astype(int)
            v = np.rint(self.camera.fy * transformed[:, 1] / safe_z + self.camera.cy).astype(int)
            inside = in_front & (u >= 0) & (u < self.camera.width) & (v >= 0) & (v < self.camera.height)
            q, nn = np.full_like(transformed, np.nan), np.full_like(transformed, np.nan)
            q[inside], nn[inside] = reference_points[v[inside], u[inside]], reference_normals[v[inside], u[inside]]
            distance = np.linalg.norm(transformed - q, axis=1)
            alignment = np.abs(np.sum((n @ transform[:3, :3].T) * nn, axis=1))
            keep = np.isfinite(nn).all(axis=1) & (distance < self.max_correspondence_m) & (alignment > .65)
            source, target_normal, target = transformed[keep], nn[keep], q[keep]
            residual = np.sum((source - target) * target_normal, axis=1)
            return source, target_normal, residual

        for iteration in range(self.max_iterations):
            source, target_normal, residual = correspondences()
            if len(source) < 30:
                return self._result("lost", "insufficient_correspondences", len(source), iterations=iteration)
            jacobian, rank, condition, covariance = self._metrics(source, target_normal, residual)
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            if rank < 6 or condition is None or condition > self.max_condition:
                return self._result("lost", "unobservable_geometry", len(source), rmse, rank, condition,
                                    iterations=iteration)
            # Huber weights retain minority surface orientations. Hard trimming
            # can erase a floor's entire constraint when most walls have zero
            # residual, incorrectly suppressing motion parallel to those walls.
            weights = np.sqrt(np.minimum(1., .03 / np.maximum(np.abs(residual), 1e-12)))
            weighted_jacobian, target = jacobian * weights[:, None], -residual * weights
            if self._gyro_used:
                # Fuse the rotation measurement throughout registration instead
                # of allowing depth noise to erase the gyro initialization.
                # The caller can supply measured body gyro noise. The angular
                # floor covers timing/calibration/integration approximation.
                gyro_std = math.hypot(self.gyro_noise_std_rps * dt, self.gyro_min_angle_std_rad)
                rotation_weight = max(rmse, .01) / gyro_std
                prior_jacobian = np.zeros((3, 6))
                prior_jacobian[:, :3] = np.eye(3) * rotation_weight
                prior_delta = _rotation_vector(rotation_prior @ transform[:3, :3].T)
                weighted_jacobian = np.vstack((weighted_jacobian, prior_jacobian))
                target = np.concatenate((target, prior_delta * rotation_weight))
            step = np.linalg.lstsq(weighted_jacobian, target, rcond=None)[0]
            if np.linalg.norm(step[:3]) > .35 or np.linalg.norm(step[3:]) > .5:
                return self._result("lost", "motion_outside_capture_range", len(source), rmse, rank, condition,
                                    iterations=iteration)
            delta = np.eye(4)
            delta[:3, :3], delta[:3, 3] = _rotation(step[:3]), step[3:]
            transform = delta @ transform
            if np.linalg.norm(step) < 1e-5:
                break
        source, target_normal, residual = correspondences()
        if len(source) < 30:
            return self._result("lost", "insufficient_correspondences", len(source))
        _, rank, condition, covariance = self._metrics(source, target_normal, residual)
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if rank < 6 or condition is None or condition > self.max_condition or rmse > self.max_rmse_m:
            return self._result("lost", "alignment_rejected", len(source), rmse, rank, condition)
        self.pose = validate_pose(self._reference_pose @ transform)
        if (np.linalg.norm(transform[:3, 3]) >= self.keyframe_translation_m or
                np.linalg.norm(_rotation_vector(transform[:3, :3])) >= self.keyframe_rotation_rad):
            self._reference, self._reference_timestamp = (points, normals), float(timestamp_s)
            self._reference_pose = self.pose.copy()
            self._gyro_rotation, self._gyro_chain_valid = np.eye(3), True
        return self._result("tracking", "depth_alignment", len(source), rmse, rank, condition,
                            covariance, iteration + 1)
