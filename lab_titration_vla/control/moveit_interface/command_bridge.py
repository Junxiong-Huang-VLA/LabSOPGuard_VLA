from __future__ import annotations

from typing import Any, Dict


def build_moveit_goal(robot_command: Dict[str, Any]) -> Dict[str, Any]:
    if robot_command.get("status") != "ready":
        return {"goal_status": "blocked", "reason": robot_command.get("reason", "not_ready")}
    targets = robot_command.get("robot_targets", {})
    return {
        "goal_status": "ready",
        "planner": "moveit2",
        "target_xyz_m": targets.get("target_xyz_m"),
        "pre_grasp_xyz_m": targets.get("pre_grasp_xyz_m"),
        "place_xyz_m": targets.get("place_xyz_m"),
        "speed_scale": robot_command.get("safety_constraints", {}).get("max_speed_scale", 0.2),
        "execution_plan": robot_command.get("execution_plan", []),
    }

