#! /usr/bin/env python

import healpy as hp #dev on 1.8.1
import numpy as np #dev on 1.8.1

class Region():
    def __init__(self,maxdepth=11):
        self.maxdepth=maxdepth
        self.pixeldict=dict( (i,set()) for i in xrange(1,maxdepth+1))
        return

    def __repr__(self):
        return "Region of with maximum depth {0}, and total area {1:5.2g}deg^2".format(self.maxdepth,self.get_area())

    def add_circles(self,ra_cen,dec_cen,radius,depth=None):
        """
        Add one or more circles to this region
        :param ra_cen: ra or list of ras for circle centers
        :param dec_cen: dec or list of decs for circle centers
        :param radius: radius or list of radii for circles
        :param depth: The depth at which we wisht to represent the circle (forced to be <=maxdepth
        :return: None
        """
        if depth==None or depth>self.maxdepth:
            depth=self.maxdepth
        try:
            ras,decs,radii = iter(ra_cen),iter(dec_cen),iter(radius)
        except TypeError:
            ras=[ra_cen]
            decs=[dec_cen]
            radii=[radius]

        for ra,dec,rad in zip(ras,decs,radii):
            pix=hp.query_disc(2**depth,self.sky2vec(ra,dec),rad,inclusive=True,nest=True)
            self.add_pixels(pix,depth)
        self._renorm()
        return

    def add_poly(self,positions):
        """
        Add a single polygon to this region
        :param positions: list of [ (ra,dec), ... ] positions that form the polygon
        :return: None
        """
        pass #as above for a polygon

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
        pd = self.pixeldict.copy()
        for d in xrange(1,self.maxdepth):
            for p in pd[d]:
                pd[d+1].update(set([4*p,4*p+1,4*p+2,4*p+3]))
        return pd[self.maxdepth]

    def _demote_all(self):
        """
        Represent this region as pixels at maxdepth only
        """
        for d in xrange(1,self.maxdepth):
            for p in self.pixeldict[d]:
                self.pixeldict[d+1].update(set([4*p,4*p+1,4*p+2,4*p+3]))
            self.pixeldict[d]=set()
        return

    def _renorm(self):
        """
        Remake the pixel dictionary, merging groups of pixels at level N into a single pixel
        at level N-1
        """
        #convert all to lowest level
        self._demote_all()
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

    def sky_within(self,ra,dec,degin=False):
        """
        Test whether a sky position is within this region
        :param ra: RA in radians
        :param dec: Dec in decimal radians
        :param degin: True if the input parameters are in degrees instead of radians
        :return: True if RA/Dec is within this region
        """
        if degin:
            theta,phi = self.sky2ang(np.radians(ra),np.radians(dec))
        else:
            theta,phi = self.sky2ang(ra,dec)
        #pixel number at the maxdepth
        pix = hp.ang2pix(2**self.maxdepth,theta,phi,nest=True)
        # print pix
        #search from shallow -> deep since shallow levels have less pixels
        for d in xrange(1,self.maxdepth+1):
            #determine the pixel number when promoted to level d
            dpix = pix//4**(self.maxdepth-d)
            # print dpix,d,self.pixeldict[d]
            if dpix in self.pixeldict[d]:
                return True
        return False

    def union(self,other,renorm=True):
        """
        Add another Region by performing union on their pixlists
        :param other: A Region
        """
        #merge the pixels that are common to both
        for d in xrange(1,min(self.maxdepth,other.maxdepth)+1):
            self.add_pixels(other.pixdict[d],d)

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
                    line="fk5; polygon "
                    x,y,z = hp.boundaries(2**d,p,step=1,nest=True)
                    for vec in zip(x,y,z):
                        #print "vec",vec
                        for ra,dec in zip(*self.vec2sky(np.array(vec),degrees=True)):
                            #print "pos",ra,dec
                            line += "{0} {1} ".format(ra,dec)
                    print>>out, line
        return

    @classmethod
    def sky2ang(cls,ra,dec):
        """
        Convert ra,dec coordinates to theta,phi coordinates
        :param ra: RA in radians
        :param dec: Dec in radians
        :return: (theta,phi)
        """
        theta=np.pi/2-dec
        phi=ra
        return theta,phi

    @classmethod
    def sky2vec(cls,ra,dec):
        """

        :param ra: RA
        :param dec: DEC
        :return: A vector list
        """
        theta_phi = cls.sky2ang(ra,dec)
        vec=hp.ang2vec(*theta_phi)
        return vec

    @classmethod
    def vec2sky(cls,vec,degrees=False):
        """
        Convert [x,y,z] vectors into sky coordinates ra,dec
        :param vec: A vector list
        :param degrees: Return ra/dec in degrees? Default = false
        :return: ra, dec in radians or degrees
        """
        theta,phi =hp.vec2ang(vec)
        ra=phi
        dec=np.pi/2-theta

        if degrees:
            ra=np.degrees(ra)
            dec=np.degrees(dec)
        return ra,dec

def test_renorm_demote():
    ra=13.5
    dec=-90
    radius=0.1
    print "RA:{0},DEC:{1}, radius:{2}".format(ra,dec,radius)
    region=Region(maxdepth=11)
    region.add_circles(np.radians(ra),np.radians(dec),np.radians(radius))
    region._demote_all()
    start_dict= region.pixeldict.copy()
    print start_dict
    region._renorm()
    region._demote_all()
    end_dict=region.pixeldict.copy()
    print end_dict

def test_sky_within():
    ra=[13.5, 15]
    dec=[-45, -40]
    radius=[0.1,0.1]
    print "RA:{0},DEC:{1}, radius:{2}".format(ra,dec,radius)
    region=Region(maxdepth=11)
    region.add_circles(np.radians(ra),np.radians(dec),np.radians(radius))
    print region.sky_within(np.radians(ra[0]),np.radians(dec[0]))
    print region.sky_within(np.radians(ra[0]+5*radius[0]),np.radians(dec[0]))

def test_conversions():
    ra=13.5
    dec=-45
    print "input",ra,dec
    vec=Region.sky2vec(np.radians(ra),np.radians(dec))
    print "vector",vec
    ra,dec=Region.vec2sky(vec)
    print "output",np.degrees(ra),np.degrees(dec)

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
    print "Pickle dump/load works =",region.pixeldict == region2.pixeldict
    return

def test_reg():
    # ra=66.38908
    # dec= -26.72466
    # radius=22
    ra=[55]
    dec=[-20]
    radius=[9]
    # #print "RA:{0},DEC:{1}, radius:{2}".format(ra,dec,radius)
    region=Region(maxdepth=9)
    region.add_circles(np.radians(ra),np.radians(dec),np.radians(radius))
    r2=Region(maxdepth=9)
    r2.add_circles(np.radians(66.389),np.radians(-26.72466),np.radians(22))
    r2.without(region)
    r2.add_circles(np.radians(ra),np.radians(dec),np.radians([1]))
    r2.write_reg('/home/hancock/temp/test.reg')


if __name__=="__main__":
    # test_renorm_demote()
    # test_sky_within()
    # test_conversions()
    # test_reg()
    # test_pickle()