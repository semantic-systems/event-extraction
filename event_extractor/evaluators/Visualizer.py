import numpy as np
import random
from abc import abstractmethod
from event_extractor.schema import FeatureToVisualize
from mayavi import mlab
from sklearn.preprocessing import normalize


class Visualizer(object):
    def __init__(self, data: FeatureToVisualize):
        self.data = data

    @abstractmethod
    def visualize(self, data: FeatureToVisualize):
        raise NotImplementedError


class SphericalVisualize(Visualizer):
    def __init__(self, data: FeatureToVisualize):
        super(SphericalVisualize, self).__init__(data)

    def visualize(self, data: FeatureToVisualize):
        # Create a sphere
        r = 1.0
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)

        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
        mlab.clf()
        normalized_catesian = self.normalize(data.feature)
        mlab.mesh(x, y, z, color=(0.0, 0.5, 0.5), opacity=0.5)
        mlab.points3d(normalized_catesian[:, 0], normalized_catesian[:, 1], normalized_catesian[:, 2],
                      [int(n) for n in data.labels], scale_mode='none',
                      scale_factor=0.05)
        mlab.show()

    def normalize(self, x: np.array):
        return normalize(x, norm="l2")


if __name__ == "__main__":
    data = FeatureToVisualize(feature=np.random.random((100, 3)), labels=[str(n) for n in list(range(0, 50))+list(range(0, 50))])
    visualizer = SphericalVisualize(data)
    visualizer.visualize(data)

