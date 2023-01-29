import numpy as np
from scipy import spatial

class IrregularGridInterpolator(object):
    def __init__(self, x0, y0, x1, y1):
        '''
        Parameters:
        -----------
        x0 : np.ndarray
            x-coords of source points
        y0 : np.ndarray
            y-coords of source points
        x1 : np.ndarray
            x-coords of destination points
        y1 : np.ndarray
            y-coords of destination points

        Sets:
        -----
        self.vertices: np.ndarray(int)
            shape = (num_target_points, 3)
        self.weights: np.ndarray(float)
            shape = (num_target_points, 3)

        Follows this suggestion:
        https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
        x_target[i] = \sum_{j=0}^2 weights[i, j]*x_source[vertices[i, j]]
        y_target[i] = \sum_{j=0}^2 weights[i, j]*y_source[vertices[i, j]]
        We can do (linear) interpolation by replacing x_target, x_source with z_target, z_source
        where z_source is the field to be interpolated and z_target is the interpolated field
        '''

        # define and triangulate source points
        self.src_shape = x0.shape
        self.src_points = np.array([x0.flatten(), y0.flatten()]).T
        self.delaunay_tri = spatial.qhull.Delaunay(self.src_points)

        # define target points
        self.dst_points = np.array([x1.flatten(), y1.flatten()]).T
        self.dst_shape = x1.shape

        # get barycentric coords
        d = 2
        self.simplex = self.delaunay_tri.find_simplex(self.dst_points)
        self.vertices = np.take(self.delaunay_tri.simplices, self.simplex, axis=0)
        temp = np.take(self.delaunay_tri.transform, self.simplex, axis=0)
        delta = self.dst_points - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)

        # set weights
        # - change negative weights to nans so that points outside the moorings grid become nans
        self.weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        self.weights[self.weights<0] = np.nan

    def interp_field(self, fld):
        '''
        Interpolate field from source points to destination points

        Parameters:
        -----------
        fld: np.ndarray
            field to be interpolated

        Returns:
        -----------
        fld_interp : np.ndarray
            field interpolated onto the destination points
        '''
        assert(fld.shape == self.src_shape)
        fld_interp = np.einsum('nj,nj->n', np.take(
            fld.flatten(),
            self.vertices),
            self.weights,
            )
        return fld_interp.reshape(self.dst_shape)
