from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def bbox_center_xyxy(bbox: List[int]) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def depth_summary(depth_value: float) -> Dict[str, float]:
    return {
        "center_depth": float(depth_value),
        "min_depth": max(float(depth_value) - 0.01, 0.0),
        "max_depth": float(depth_value) + 0.01,
    }


def pixel_to_camera_xyz(
    u: float,
    v: float,
    z_m: float,
    intrinsics: Optional[Dict[str, float]] = None,
) -> Optional[List[float]]:
    if z_m <= 0.0 or not intrinsics:
        return None
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError):
        return None
    if fx == 0.0 or fy == 0.0:
        return None
    x = (float(u) - cx) * z_m / fx
    y = (float(v) - cy) * z_m / fy
    return [x, y, z_m]


def _extract_matrix_extrinsics(extrinsics: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    rotation = extrinsics.get("rotation")
    translation = extrinsics.get("translation")
    if isinstance(rotation, list) and isinstance(translation, list) and len(rotation) == 9 and len(translation) == 3:
        return [float(v) for v in rotation], [float(v) for v in translation]
    return None, None


def _extract_quaternion_extrinsics(
    extrinsics: Dict[str, Any],
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    # ROS style: {transform: {translation:{x,y,z}, rotation:{x,y,z,w}}}
    transform = extrinsics.get("transform")
    if isinstance(transform, dict):
        t = transform.get("translation", {})
        q = transform.get("rotation", {})
    else:
        # flat fallback: {translation:{x,y,z}, rotation:{x,y,z,w}}
        t = extrinsics.get("translation", {})
        q = extrinsics.get("rotation", {})

    if not isinstance(t, dict) or not isinstance(q, dict):
        return None, None
    req_t = ("x", "y", "z")
    req_q = ("x", "y", "z", "w")
    if any(k not in t for k in req_t) or any(k not in q for k in req_q):
        return None, None

    translation = [float(t["x"]), float(t["y"]), float(t["z"])]
    quaternion = [float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])]
    return quaternion, translation


def _rotate_by_quaternion(point_xyz: List[float], q_xyzw: List[float]) -> List[float]:
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    qx, qy, qz, qw = [float(v) for v in q_xyzw]
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm == 0.0:
        return [x, y, z]
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Quaternion to rotation matrix
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    xr = r00 * x + r01 * y + r02 * z
    yr = r10 * x + r11 * y + r12 * z
    zr = r20 * x + r21 * y + r22 * z
    return [xr, yr, zr]


def camera_to_robot_xyz(
    camera_xyz: Optional[List[float]],
    extrinsics: Optional[Dict[str, Any]] = None,
) -> Optional[List[float]]:
    if camera_xyz is None:
        return None
    if not extrinsics:
        return camera_xyz

    # 1) matrix form
    rot_m, trans_m = _extract_matrix_extrinsics(extrinsics)
    if rot_m is not None and trans_m is not None:
        x, y, z = float(camera_xyz[0]), float(camera_xyz[1]), float(camera_xyz[2])
        xr = rot_m[0] * x + rot_m[1] * y + rot_m[2] * z + trans_m[0]
        yr = rot_m[3] * x + rot_m[4] * y + rot_m[5] * z + trans_m[1]
        zr = rot_m[6] * x + rot_m[7] * y + rot_m[8] * z + trans_m[2]
        return [xr, yr, zr]

    # 2) quaternion form (ROS style)
    quat, trans_q = _extract_quaternion_extrinsics(extrinsics)
    if quat is not None and trans_q is not None:
        xr, yr, zr = _rotate_by_quaternion(camera_xyz, quat)
        return [xr + trans_q[0], yr + trans_q[1], zr + trans_q[2]]

    # fallback to camera frame if extrinsics schema is unknown
    return camera_xyz

