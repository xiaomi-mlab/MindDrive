import numpy as np

# ==============================================================================
# -- import route planning (copied and modified from CARLA 0.9.4's PythonAPI) --
# ==============================================================================
import carla
from rl_projects.envs.carla_env.navigation.local_planner import RoadOption
from rl_projects.envs.carla_env.navigation.global_route_planner import GlobalRoutePlanner
from rl_projects.envs.carla_env.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from rl_projects.envs.carla_env.misc import vector

def compute_route_waypoints(world_map, start_waypoint, end_waypoint, resolution=1.0, plan=None):
    """
        Returns a list of (waypoint, RoadOption)-tuples that describes a route
        starting at start_waypoint, ending at end_waypoint.

        start_waypoint (carla.Waypoint):
            Starting waypoint of the route
        end_waypoint (carla.Waypoint):
            Destination waypoint of the route
        resolution (float):
            Resolution, or lenght, of the steps between waypoints
            (in meters)
        plan (list(RoadOption) or None):
            If plan is not None, generate a route that takes every option as provided
            in the list for every intersections, in the given order.
            (E.g. set plan=[RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.RIGHT]
            to make the route go straight, then left, then right.)
            If plan is None, we use the GlobalRoutePlanner to find a path between
            start_waypoint and end_waypoint.
    """

    if plan is None:
        # Setting up global router
        dao = GlobalRoutePlannerDAO(world_map, resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        
        # Obtain route plan
        route = grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)
    else:
        # Compute route waypoints
        route = []
        current_waypoint = start_waypoint
        for i, action in enumerate(plan):
            # Generate waypoints to next junction
            wp_choice = [current_waypoint]
            while len(wp_choice) == 1:
                current_waypoint = wp_choice[0]
                route.append((current_waypoint, RoadOption.LANEFOLLOW))
                wp_choice = current_waypoint.next(resolution)

                # Stop at destination
                if i > 0 and current_waypoint.transform.location.distance(end_waypoint.transform.location) < resolution:
                    break

            if action == RoadOption.VOID:
                break

            # Make sure that next intersection waypoints are far enough
            # from each other so we choose the correct path
            step = resolution
            while len(wp_choice) > 1:
                wp_choice = current_waypoint.next(step)
                wp0, wp1 = wp_choice[:2]
                if wp0.transform.location.distance(wp1.transform.location) < resolution:
                    step += resolution
                else:
                    break

            # Select appropriate path at the junction
            if len(wp_choice) > 1:
                # Current heading vector
                current_transform = current_waypoint.transform
                current_location = current_transform.location
                projected_location = current_location + \
                    carla.Location(
                        x=np.cos(np.radians(current_transform.rotation.yaw)),
                        y=np.sin(np.radians(current_transform.rotation.yaw)))
                v_current = vector(current_location, projected_location)

                direction = 0
                if action == RoadOption.LEFT:
                    direction = 1
                elif action == RoadOption.RIGHT:
                    direction = -1
                elif action == RoadOption.STRAIGHT:
                    direction = 0
                select_criteria = float("inf")

                # Choose correct path
                for wp_select in wp_choice:
                    v_select = vector(
                        current_location, wp_select.transform.location)
                    cross = float("inf")
                    if direction == 0:
                        cross = abs(np.cross(v_current, v_select)[-1])
                    else:
                        cross = direction * np.cross(v_current, v_select)[-1]
                    if cross < select_criteria:
                        select_criteria = cross
                        current_waypoint = wp_select

                # Generate all waypoints within the junction
                # along selected path
                route.append((current_waypoint, action))
                current_waypoint = current_waypoint.next(resolution)[0]
                while current_waypoint.is_intersection:
                    route.append((current_waypoint, action))
                    current_waypoint = current_waypoint.next(resolution)[0]
        assert route

    # Change action 5 wp before intersection
    num_wp_to_extend_actions_with = 5
    action = route[0][1]
    for i in range(1, len(route)):
        next_action = route[i][1]
        if next_action != action:
            if next_action != RoadOption.LANEFOLLOW:
                for j in range(num_wp_to_extend_actions_with):
                    route[i-j-1] = (route[i-j-1][0], route[i][1])
        action = next_action

    return route
