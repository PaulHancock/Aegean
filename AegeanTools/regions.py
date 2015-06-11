#! /usr/bin/env python

import os
import datetime
import healpy as hp #dev on 1.8.1
import numpy as np #dev on 1.8.1
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits


class Region():
    def __init__(self,maxdepth=11):
        self.maxdepth=maxdepth
        self.pixeldict=dict( (i,set()) for i in xrange(1,maxdepth+1))
        self.demoted=[]
        return

    def __repr__(self):
        return "Region of with maximum depth {0}, and total area {1:5.2g}deg^2".format(self.maxdepth,self.get_area())

    def add_circles(self,ra_cen,dec_cen,radius,depth=None):
        """
        Add one or more circles to this region
        :param ra_cen: ra or list of ras for circle centers
        :param dec_cen: dec or list of decs for circle centers
        :param radius: radius or list of radii for circles
        :param depth: The depth at which we wish to represent the circle (forced to be <=maxdepth
        :return: None
        """
        if depth==None or depth>self.maxdepth:
            depth=self.maxdepth
        try:
            sky=zip(ra_cen,dec_cen)
            rad=radius
        except TypeError:
            sky= [[ra_cen,dec_cen]]
            rad=[radius]
        vectors = self.sky2vec(sky)
        for vec,r in zip(vectors,rad):
            pix=hp.query_disc(2**depth,vec,r,inclusive=True,nest=True)
            self.add_pixels(pix,depth)
        self._renorm()
        return

    def add_poly(self,positions,depth=None):
        """
        Add a single polygon to this region
        :param positions: list of [ (ra,dec), ... ] positions that form the polygon
        :param depth: The depth at which we wish to represent the circle (forced to be <=maxdepth
        :return: None
        """
        assert len(positions)>=3, "A minimum of three coordinate pairs are required"

        if depth==None or depth>self.maxdepth:
            depth=self.maxdepth

        ras,decs =zip(*positions)
        sky=self.radec2sky(ras,decs)
        pix=hp.query_polygon(2**depth,self.sky2vec(sky),inclusive=True,nest=True)
        self.add_pixels(pix,depth)
        self._renorm()
        return

    def add_pixels(self,pix,depth):
        if depth not in self.pixeldict:
            self.pixeldict[depth]=set()
        self.pixeldict[depth].update(set(pix))
        pass

    def get_area(self,degrees=True):
        area=0
        for d in xrange(1,self.maxdepth+1):
            area+=len(self.pixeldict[d])*hp.nside2pixarea(2**d,degrees=degrees)
        return area

    def get_demoted(self):
        """
        :return: Return a set of pixels that represent this region at maxdepth
        """
        self._demote_all()
        return self.demoted

    def _demote_all(self):
        """
        Represent this region as pixels at maxdepth only
        """
        pd = self.pixeldict.copy()
        for d in xrange(1,self.maxdepth):
            for p in pd[d]:
                pd[d+1].update(set((4*p,4*p+1,4*p+2,4*p+3)))
        self.demoted = list(pd[d+1])
        return

    def _renorm(self):
        """
        Remake the pixel dictionary, merging groups of pixels at level N into a single pixel
        at level N-1
        """
        #convert all to lowest level
        self._demote_all()
        #store this for later
        self.demoted=self.pixeldict
        #now promote as needed
        for d in xrange(self.maxdepth,2,-1):
            plist=self.pixeldict[d].copy()
            for p in plist:
                if p%4==0:
                    nset=set((p,p+1,p+2,p+3))
                    if p+1 in plist and p+2 in plist and p+3 in plist:
                    # if nset.intersection(plist) != set():
                        #remove the four pixels from this level
                        self.pixeldict[d].difference_update(nset)
                        #add a new pixel to the next level up
                        self.pixeldict[d-1].add(p/4)
        return

    #@profile
    def sky_within(self,ra,dec,degin=False):
        """
        Test whether a sky position is within this region
        :param ra: RA in radians
        :param dec: Dec in decimal radians
        :param degin: True if the input parameters are in degrees instead of radians
        :return: True if RA/Dec is within this region
        """
        sky=self.radec2sky(ra,dec)

        if degin:
            sky=np.radians(sky)

        theta_phi = self.sky2ang(sky)
        theta,phi = theta_phi.transpose()
        pix = hp.ang2pix(2**self.maxdepth,theta,phi,nest=True)
        pixelset = self.get_demoted()
        result = np.in1d(pix,list(pixelset))
        return result

    def union(self,other,renorm=True):
        """
        Add another Region by performing union on their pixlists
        :param other: A Region
        """
        #merge the pixels that are common to both
        for d in xrange(1,min(self.maxdepth,other.maxdepth)+1):
            self.add_pixels(other.pixeldict[d],d)

        #if the other region is at higher resolution, then include a degraded version of the remaining pixels.
        if self.maxdepth<other.maxdepth:
            for d in xrange(self.maxdepth+1,other.maxdepth+1):
                for p in other.pixeldict[d]:
                    #promote this pixel to self.maxdepth
                    pp = p/4**(d-self.maxdepth)
                    self.pixeldict[self.maxdepth].add(pp)
        if renorm:
            self._renorm()
        return

    def without(self,other):
        """
        Remove the overlap between this region and the other region
        :param other: Another region
        :return: None
        """
        #work only on the lowest level
        #TODO: Allow this to be done for regions with different depths.
        assert self.maxdepth==other.maxdepth, "Regions must have the same maxdepth"
        self._demote_all()
        opd = other.get_demoted()
        self.pixeldict[self.maxdepth].difference_update(opd)
        self._renorm()
        return

    def write_reg(self,filename):
        """
        Write a ds9 region file that represents this region as a set of diamonds.
        :param filename: file to write
        :return: None
        """
        with open(filename,'w') as out:
            for d in xrange(1,self.maxdepth+1):
                for p in self.pixeldict[d]:
                    line="fk5; polygon("
                    #the following int() gets around some problems with np.int64 that exist prior to numpy v 1.8.1
                    vectors = zip(*hp.boundaries(2**d,int(p),step=1,nest=True))
                    positions=[]
                    for sky in self.vec2sky(np.array(vectors),degrees=True):
                        ra, dec = sky
                        pos= SkyCoord(ra/15,dec,unit=(u.degree,u.degree))
                        positions.append( pos.ra.to_string(sep=':',precision=2))
                        positions.append( pos.dec.to_string(sep=':',precision=2))
                    line += ','.join(positions)
                    line += ")"
                    print>>out, line
        return

    def write_fits(self, filename, moctool=''):
        """

        :param self:
        :param filename:
        :return:
        """
        datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','MOC.fits')
        hdulist = fits.open(datafile)
        cols=fits.Column(name='NPIX',array = self._uniq(),format='1K')
        tbhdu = fits.BinTableHDU.from_columns([cols])
        hdulist[1] = tbhdu
        hdulist[1].header['PIXTYPE'] = ('HEALPIX ', 'HEALPix magic code')
        hdulist[1].header['ORDERING'] = ('NUNIQ ','NUNIQ coding method')
        hdulist[1].header['COORDSYS']= ('C ','ICRS reference frame')
        hdulist[1].header['MOCORDER']= (self.maxdepth,'MOC resolution (best order)')
        hdulist[1].header['MOCTOOL'] = (moctool,'Name of the MOC generator')
        hdulist[1].header['MOCTYPE'] = ('CATALOG','Source type (IMAGE or CATALOG)')
        hdulist[1].header['MOCID'] = (' ','Identifier of the collection')
        hdulist[1].header['ORIGIN'] = (' ','MOC origin')
        time = datetime.datetime.utcnow()
        hdulist[1].header['DAATE'] = (datetime.datetime.strftime(time,format="%Y-%m-%dT%H:%m:%SZ"),'MOC creation date')
        hdulist.writeto(filename, clobber=True)
        return

    def _uniq(self):
        """

        :return:
        """
        pd = []
        for d in xrange(1,self.maxdepth):
            fn = lambda x: int(4**(d+1) + x)
            pd.extend(map(fn, self.pixeldict[d]))
        return sorted(pd)

    @classmethod
    def radec2sky(cls,ra,dec):
        """

        :param ra:
        :param dec:
        :return:
        """
        sky=np.empty((len(ra),2),dtype=type(ra[0]))
        sky[:,0]=ra
        sky[:,1]=dec
        return sky

    @classmethod
    def sky2ang(cls,sky):
        """
        Convert ra,dec coordinates to theta,phi coordinates
        ra -> phi
        dec -> theta
        :param sky: float [(ra,dec),...]
        :return: A list of [(theta,phi), ...]
        """
        try:
            theta_phi = sky.copy()
        except AttributeError, e:
            theta_phi = np.array(sky)
        theta_phi[:,[1,0]] = theta_phi[:,[0,1]]
        theta_phi[:,0] = np.pi/2 -theta_phi[:,0]
        return theta_phi

    @classmethod
    def sky2vec(cls,sky):
        """
        Convert sky positions in to 3d-vectors
        :param sky: [(ra,dec), ...]
        :return: [(x,y,z), ...]
        """
        theta_phi = cls.sky2ang(sky)
        theta,phi = map(np.array,zip(*theta_phi))
        vec=hp.ang2vec(theta,phi)
        return vec

    @classmethod
    def vec2sky(cls,vec,degrees=False):
        """
        Convert [x,y,z] vectors into sky coordinates ra,dec
        :param vec: An array-like list of ([x,y,z],...)
        :param degrees: Return ra/dec in degrees? Default = false
        :return: [(ra,...),(dec,...)]
        """
        theta,phi = hp.vec2ang(vec)
        ra=phi
        dec=np.pi/2-theta

        if degrees:
            ra=np.degrees(ra)
            dec=np.degrees(dec)
        return cls.radec2sky(ra,dec)

def test_radec2sky():
    ra,dec = (15,-45)
    sky = Region.radec2sky(ra,dec)
    assert sky == [[ra,dec]], "radec2sky broken on non-list input"
    ra = [0,10]
    dec = [-45,45]
    sky = Region.radec2sky(ra,dec)
    answer = np.array([(ra[0],dec[0]),(ra[1],dec[1])])
    assert np.all(sky == answer), 'radec2sky broken on list input'

def test_sky2ang_symmetric():
    sky = np.radians(np.array([[15,-45]]))
    tp = Region.sky2ang(sky)
    tp = np.array([ [tp[0][1],tp[0][0]]])
    sky2 = Region.sky2ang(tp)
    sky2 = np.array([ [sky2[0][1],sky2[0][0]]])
    assert np.all(abs(sky-sky2)<1e-9), "sky2ang failed to be symmetric"
    return

def test_sky2ang_corners():
    corners=np.radians([[0,0],[360,-90]])
    theta_phi = Region.sky2ang(corners)
    answers = np.array([ [np.pi/2,0],[np.pi,2*np.pi]])
    assert np.all(theta_phi - answers< 1e-9), 'sky2ang corner cases failed'

def test_sky2vec_corners():
    sky = np.radians(   [ [0,0],   [90,90], [45,-90]])
    answers = np.array( [ [1,0,0], [0,0,1], [0,0,-1]])
    vec = Region.sky2vec(sky)
    assert np.all(vec - answers<1e-9), 'sky2vec corner cases failed'

def test_vec2sky_corners():
    vectors = np.array( [ [1,0,0], [0,0,1], [0,0,-1]])
    skycoords = Region.vec2sky(vectors,degrees=True)
    answers = np.array( [ [0,0],   [0,90], [0,-90]] )
    assert np.all( skycoords == answers), 'vec2sky fails on corners'

def test_sky2vec2sky():
    ra,dec=np.radians(np.array((0,-45)))
    sky= Region.radec2sky(ra,dec)
    vec = Region.sky2vec(sky)
    sky2 = Region.vec2sky(vec)
    assert np.all(np.array(sky2) - np.array(sky) ==0 ), "sky2vec2sky failed"
    vec2 = Region.sky2vec(sky2)
    assert np.all(np.array(vec) - np.array(vec2) ==0), 'vec2sky2vec failed'

def test_add_circles_list_scalar():
    ra_list = np.radians([13.5,13.5])
    dec_list = np.radians([-90,-90])
    radius_list = np.radians([0.1,0.01])
    ra = ra_list[0]
    dec = dec_list[0]
    radius = radius_list[0]
    region1=Region(maxdepth=11)
    region2=Region(maxdepth=11)
    region1.add_circles(ra_list,dec_list,radius_list)
    region1._demote_all()
    region2.add_circles(ra,dec,radius)
    region2._demote_all()
    test=True
    for i in xrange(1,region1.maxdepth+1):
        if len(region1.pixeldict[i].difference(region2.pixeldict[i])) >0:
            test=False
    assert test, 'add_circles gives different results for lists and scalars'

def test_renorm_demote_symmetric():
    ra=13.5
    dec=-90
    radius=0.1
    #print "RA:{0},DEC:{1}, radius:{2}".format(ra,dec,radius)
    region=Region(maxdepth=11)
    region.add_circles(np.radians(ra),np.radians(dec),np.radians(radius))
    region._demote_all()
    start_dict=region.pixeldict.copy()
    region._renorm()
    region._demote_all()
    end_dict=region.pixeldict.copy()
    test=True
    for i in xrange(1,region.maxdepth+1):
        if len(end_dict[i].difference(start_dict[i])) >0:
            test=False
    assert test, 'renorm and demote are not symmetric'

def test_sky_within():
    ra=np.radians([13.5, 15])
    dec=np.radians([-45, -40])
    radius=np.radians([0.1,0.1])
    region=Region(maxdepth=11)
    region.add_circles(ra,dec,radius)
    assert np.all(region.sky_within(ra[0],dec[0])), "Failed on position at center of region"
    assert np.all(region.sky_within(ra,dec)), "Failed on list of positions"
    assert not np.any(region.sky_within(ra[0]+5*radius[0],dec[0])), "Failed on position outside of region"

def test_pickle():
    ra=66.38908
    dec= -26.72466
    radius=22
    region=Region(maxdepth=8)
    region.add_circles(np.radians(ra),np.radians(dec),np.radians(radius))
    try:
        import cPickle as pickle
    except:
        import pickle
    pickle.dump(region,open('out.mim','w'))
    region2=pickle.load(open('out.mim'))
    assert region.pixeldict == region2.pixeldict, 'pickle/unpickle does not give same region'
    return

def test_reg():
    ra=np.radians([285])
    dec=np.radians([-66])
    radius=np.radians([3])
    region=Region(maxdepth=9)
    region.add_circles(ra,dec,radius)
    region.write_reg('test.reg')

def test_poly():
    ra=[50,50,70,70]
    dec=[-20,-25,-25,-20]
    region=Region(maxdepth=9)
    positions=zip(np.radians(ra),np.radians(dec))
    region.add_poly(positions)
    region.write_reg('test.reg')

def test_write_fits():
    a = Region()
    a.add_circles(12,0,0.1)
    a.write_fits('test_MOC.fits')
    return

if __name__=="__main__":
    # test_vec2sky_corners()
    # test_reg()
    # test_radec2sky()
    # test_sky2ang_symmetric()
    # test_sky2ang_corners()
    # test_sky2vec2sky()
    # test_poly()
    # test_renorm_demote_symmetric()
    # test_add_circles_list_scalar()
    # test_sky_within()
    # test_pickle()
    # test_sky2vec_corners()
    test_write_fits()