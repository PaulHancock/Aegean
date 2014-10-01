#! /usr/bin/env python

import healpy as hp
import numpy as np

nside=2**11 #gives pixels of approx 1.7x1.7 arcmin

class Region():
    def __init__(self,maxdepth=11):
        self.maxdepth=maxdepth
        self.pixeldict=dict( (i,set()) for i in xrange(1,maxdepth+1))
        return

    def add_circles(self,ra_cen,dec_cen,radius,depth=None):
        """
        Add a set of circles to this region
        :param ra_cen:
        :param dec_cen:
        :param radius:
        :param depth:
        :return:
        """
        if depth==None:
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
        pass #as above for a polygon

    def get_area(self):
        pass

    def get_pixeldict(self):
        return self.pixeldict

    def add_pixels(self,pix,depth):
        if depth not in self.pixeldict:
            self.pixeldict[depth]=set()
        self.pixeldict[depth].update(set(pix))
        pass

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
                    if nset.intersection(plist) != set():
                        #remove the four pixels from this level
                        self.pixeldict[d].difference_update(nset)
                        #add a new pixel to the next level up
                        self.pixeldict[d-1].add(p/4)
        return

    def sky_within(self,ra,dec):
        """
        Test whether a sky position is within this region
        :param ra: RA in radians
        :param dec: Dec in decimal radians
        :return: True if RA/Dec is within this region
        """

        theta,phi = self.sky2ang(ra,dec)
        #pixel number at the maxdepth
        pix = hp.ang2pix(2**self.maxdepth,theta,phi,nest=True)
        print pix
        #search from shallow -> deep since shallow levels have less pixels
        for d in xrange(1,self.maxdepth+1):
            #determine the pixel number when promoted to level d
            dpix = pix//4**(self.maxdepth-d)
            print dpix,d,self.pixeldict[d]
            if dpix in self.pixeldict[d]:
                return True
        return False

    def union(self,other,renorm=True):
        """
        Add another Region by performing union on their pixlists
        :param other: A Region
        """
        for d in xrange(1,max(self.maxdepth,other.maxdepth)+1):
            self.add_pixels(other.pixdict[d],d)
        if renorm:
            self._renorm()
        return

    def difference(self,other):
        pass #as above, propagating set operations

    def area(self,degrees=True):
        return len(self.pixel)*hp.nside2pixarea(self.nside,degrees=degrees)

    def __repr__(self):
        return "Region of with maximum depth {0}".format(self.maxdepth)

    @classmethod
    def sky2ang(self,ra,dec):
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
    def sky2vec(self,ra,dec):
        """

        :param ra:
        :param dec:
        :return:
        """
        x,y,z=hp.ang2vec(*self.sky2ang(ra,dec))
        return x,y,z

def test_renorm_demote():
    ra=13.5
    dec=-90
    radius=0.1
    print "RA:{0},DEC:{1}, radius:{2}".format(ra,dec,radius)
    region=Region(maxdepth=11)
    region.add_circles(np.radians(ra),np.radians(dec),np.radians(radius))
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


if __name__=="__main__":
    #test_renorm_demote()
    test_sky_within()

#return the 8 nearest neighbouring pixels
#get_all_neighbours(nside,theta,phi)
#get_all_neighbours(nside,pix)

#as above but not including diagonals
#get_neighbours(nside,theta,phi)

#convert an angle on the sky to a healpix pixel
#ang2pix( 4, [theta,phi])

#return all ipix that are within the given region
#region specified as verticies (not ra/dec)
#hp.query_polygon(4, [ [0,1,1],[1,1,1],[0,0,1]])

#return all pix that are within a disk
# if inclusive then return all pix that overlap the disc
# else include only those whose centers are within the disc
#hp.query_disc(4,vec,radius,inclusive=True)

#convert theta,phi to a vector
# dec = pi/2-theta
# ra = phi
# both in radians
#hp.ang2vec(theta,phi)

#to check if a position is within a given pixel we just check
#if ang2pix(nside, [theta,phi]) == pix
