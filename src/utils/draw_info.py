import pybullet as p


def draw_trajectory(p, ee_pos, trajectory_point_ids):
    for i in range(34):
        p.resetBasePositionAndOrientation(trajectory_point_ids[i], ee_pos[i], [0,0,0,1])


def draw_pick_and_place_area(pick_area, place_area):
    # draw the area as rectangle in pybullet
    p.addUserDebugLine(pick_area[0], pick_area[1], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[1], pick_area[3], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[3], pick_area[2], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[2], pick_area[0], lineColorRGB=[1, 0, 0], lineWidth=2)

    p.addUserDebugLine(place_area[0], place_area[1], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[1], place_area[3], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[3], place_area[2], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[2], place_area[0], lineColorRGB=[0, 1, 0], lineWidth=2)