from mpscenes.obstacles.box_obstacle import BoxObstacle


movable_obstacle_dict = {
    'type': 'box',
    'geometry': {
        'position' : [0.3, 0.3, 0.02], # x, y, z | Note that it must have the (height / 2) of the box as z value to avoid weird behaviour.
        'width': 0.04,
        'height': 0.04,
        'length': 0.1,
    },
    'movable': True,
    'high': { 
        'position' : [0, 0, 1.0],
        'width': 0.35,
        'height': 0.35,
        'length': 0.35,
    },
    'low': {
        'position' : [0.0, 0.0, 0.5],
        'width': 0.2,
        'height': 0.2,
        'length': 0.2,
    }
}
movable_obstacle = BoxObstacle(name="movable_box", content_dict=movable_obstacle_dict)