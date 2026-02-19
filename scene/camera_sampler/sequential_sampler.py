class SequentialSampler:
    def __init__(self, train_cameras, settings=dict()):
        self.train_cameras = train_cameras
        self.viewpoint_stack = None

    def get_camera(self, current_batch):
        # Wrong but used only for tests. Do not use it in normal training
        return self.train_cameras[current_batch]
