from random import randint


class RandomSampler:
    def __init__(self, train_cameras, settings=dict()):
        self.train_cameras = train_cameras
        self.viewpoint_stack = None

    def get_camera(self, current_batch):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.train_cameras.copy()
        camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        return camera

    def update(self, key: str, value):
        return
