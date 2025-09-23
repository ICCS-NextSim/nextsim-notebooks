import numpy as np
import pyproj
import cartopy.crs as ccrs

class ProjectionInfo:

    def __init__(self,
            ecc    = 0.081816153,
            a      = 6378.273e3,
            lat_0  = 90.,
            lon_0  = -45.,
            lat_ts = 60.,
            proj='stere',
            ):
        '''
        Default is the projection used by neXtSIM

        Parameters:
        -----------
        proj : str
            choices 'stere' (stereographic projection)
            or 'laea' (Lambert azimuthal equal area)
        ecc : float
            eccentricity of ellipsoid for globe
        a : float
            semi-major axis of ellipsoid for globe (radius at equator)
        lat_0 : float
            central latitude
        lon_0 : float
            central longitude
        lat_ts : float
            true scale latitude
        '''
        assert(proj in ['stere', 'laea'])
        self.proj   = proj
        self.ecc    = ecc
        self.a      = a #equatorial radius in m
        self.lat_0  = lat_0
        self.lon_0  = lon_0
        self.lat_ts = lat_ts

    @classmethod
    def init_from_mppfile(cls, mppfile=None):

        if mppfile is None:
            return ProjectionInfo()

        # init self and read the mppfile
        self = cls.__new__(cls) # empty object
        super(cls, self).__init__()
        with open(mppfile, 'r') as mf:
            lines = mf.readlines()

        # stere info
        self.proj = 'stere'
        (self.lat_0,
         lon_pole,
         self.lat_ts) = np.array(
                 lines[1].split()[:3],
                 dtype=float)
        self.lon_0 = float(lines[2].split()[0]) # rotation

        # shape of earth
        scale = float(lines[3].split()[0])
        self.ecc = float(lines[-1].split()[0])
        self.a = float(lines[-2].split()[0])/scale # convert to m

        return self

    @classmethod
    def osisaf_nsidc_np_stere(cls):
        '''
        * Used by Tian-Kunze for AMSR2 and SMOS
        * Also same projection seems to be used by OSISAF
        * Hughes ellipsoid
        * get from ftp://ftp-projects.cen.uni-hamburg.de/seaice/AMSR2/README.txt
          or https://nsidc.org/data/polar-stereo/ps_grids.html with some manual inspection
        '''
        return ProjectionInfo(
                proj='stere',
                a=6378273,
                ecc=0.081816153,
                lat_0=90,
                lat_ts=70,
                lon_0=90, # from inspection
                )

    @classmethod
    def osisaf_nsidc_sp_stere(cls):
        '''
        * Used by nsidc/osisaf stere for the southern ocean
        '''
        return ProjectionInfo(
                proj='stere',
                a=6378273,
                ecc=0.081816153,
                lat_0=-90,
                lat_ts=-70,
                lon_0=0, # from inspection
                )

    @classmethod
    def nextsim_south(cls):
        '''
        * Used by neXtSIM for the southern ocean
        '''
        return ProjectionInfo(
                proj='stere',
                a=6378273,
                ecc=0.081816153,
                lat_0=-90,
                lat_ts=-60,
                lon_0=0, # from inspection
                )

    @classmethod
    def topaz_np_stere(cls):
        '''
        * info in ncfile is incomplete
        * work out projection from partial info & hyc2proj source code
          https://github.com/nansencenter/NERSC-HYCOM-CICE/blob/master/hycom/MSCPROGS/src/Hyc2proj/mod_toproj.F90
        '''
        return ProjectionInfo(
                proj='stere',
                a=6378273, ecc=0.,
                lat_0=90, lat_ts=90, lon_0=-45)

    @classmethod
    def np_laea(cls):
        '''
        * Lambert azimuthal equal area projection on WGS84 ellipse
        * used by CS2-SMOS
        '''
        a = 6378137
        b = 6356752.3142
        return ProjectionInfo(
                proj='laea',
                lon_0=0, lat_0=90,
                a=a, ecc = cls.get_eccentricity(a, b))

    @classmethod
    def sp_laea(cls):
        '''
        * Lambert azimuthal equal area projection on WGS84 ellipse
        * used by OSISAF-EASE SH
        '''
        a = 6378137
        b = 6356752.3142
        return ProjectionInfo(
                proj='laea',
                lon_0=0, lat_0=-90,
                a=a, ecc = cls.get_eccentricity(a, b))
    @classmethod
    def get_eccentricity(cls, a, b):
        '''
        Get eccentricity from a, b (major and minor axes of earth ellipsoids)

        Parameters:
        -----------
        a : float
            major semi-axis of ellipsoid 
        b : float
            minor semi-axis of ellipsoid 

        Returns:
        --------
        ecc: float
            eccentricity = 1\sqrt{1-(b/a)^2}
        '''
        assert(b<=a)
        return np.sqrt( 1. - (float(b)/a)**2 )

    @property
    def b(self):
        '''
        Returns:
        --------
        b : float
            minor semi-axis of ellipsoid
        '''
        return self.a * np.sqrt( 1 - self.ecc**2)

    @property
    def k_0(self):
        '''
        Returns:
        -------
        k_0 : float
            scale factor at origin
            See https://www.unidata.ucar.edu/software/netcdf-java/current/reference/StandardCoordinateTransforms.html
        '''
        sin = np.abs(np.sin(np.deg2rad(self.lat_ts)))
        return (1 + sin)/2

    @property
    def earthshape(self):
        return self.a, self.b

    @property
    def geodesic(self):
        return pyproj.Geod(a=self.a, b=self.b)

    @property
    def pyproj(self):
        return pyproj.Proj(proj=self.proj,
                a=self.a, b=self.b,
                lon_0=self.lon_0, lat_0=self.lat_0, lat_ts=self.lat_ts)

    @property
    def crs(self):
        if self.proj=='stere':
            return ccrs.Stereographic(
                                  central_latitude=self.lat_0,
                                  central_longitude=self.lon_0,
                                  true_scale_latitude=self.lat_ts,
                                  globe=self.globe)
        elif self.proj=='laea':
            return ccrs.LambertAzimuthalEqualArea(
                                  central_latitude=self.lat_0,
                                  central_longitude=self.lon_0,
                                  globe=self.globe)

    @property
    def globe(self):
        return ccrs.Globe(semimajor_axis=self.a,
                semiminor_axis=self.b)

    def __str__(self):
        """
        print projection info
        """
        sep = "\n"
        return sep.join([
                self.proj,
                str(self.a),
                str(self.b),
                str(self.lat_0),
                str(self.lon_0),
                str(self.lat_ts)
                ])

    def ncattrs(self, grid_mapping_name):
        '''
        Get the netcdf attributes to set for a netcdf projection variable.
        See https://www.unidata.ucar.edu/software/netcdf-java/current/reference/StandardCoordinateTransforms.html

        Parameters:
        -----------
        grid_mapping_name : str

        Returns:
        --------
        ncatts : dict
        '''
        if self.proj != 'stere':
            raise ValueError("ncattrs only implemented for proj='stere'")
        return dict(
            latitude_of_projection_origin = float(self.lat_0),
            longitude_of_projection_origin = float(self.lon_0),
            straight_vertical_longitude_from_pole = float(self.lon_0),
            semi_major_axis = float(self.a),
            semi_minor_axis = float(self.b),
            scale_factor_at_projection_origin=float(self.k_0),
            grid_mapping_name = grid_mapping_name,
            false_northing = float(0.),
            false_easting = float(0.),
            proj4 = self.pyproj.srs,
            )
