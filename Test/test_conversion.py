import aegean as a
import random
from glob import glob
import sys

if len(sys.argv)>1:
    ntest =int(sys.argv[-1])
else:
    ntest = 10


for filename in glob('Test/Images/*.fits'):
    img = a.FitsImage(filename)
    print "File {0}".format(filename)
    global_data = a.GlobalFittingData
    global_data.wcs = img.wcs
    global_data.hdu_header= img.get_hdu_header()
    a.global_data = global_data


    #print "TESTING sky2pix and pix2sky"
    ypix,xpix = img.get_pixels().shape
    for x,y in [(random.randint(0,xpix),random.randint(0,ypix)) for i in range(ntest)]:
        ra,dec = a.pix2sky((x,y))
        xout,yout = a.sky2pix((ra,dec))
        diffx,diffy = abs(x-xout),abs(y-yout)
        if not (diffx<1e-6 and diffy<1e-6):
            print filename,"{0:3d} {1:3d} -> {2:5.3e} {3:5.3e} -> {4:3d} {5:3d} | {6:5.3e} {7:5.3e} FAIL".format(x,y,ra,dec,int(xout),int(yout), diffx,diffy)
            break

    #print "TESTING sky2pix_vec and pix2sky_vec"
    for major,ratio,pa in [(random.random()*5e-2,0.1+random.random()*0.9,random.randint(-90,90)) for i in range(ntest)]:
        minor=major*ratio
        beam = a.Beam(major,minor,pa)
        pixbeam1 = a.get_pixbeam(beam,0,0)
        major_out,pa_out = a.pix2sky_vec([0,0],pixbeam1.a,pixbeam1.pa)[2:]
        minor_out,junk = a.pix2sky_vec([0,0],pixbeam1.b,pixbeam1.pa+90)[2:]
        diffmaj=abs(beam.a-major_out)
        diffmin=abs(beam.b-minor_out)
        diffpa=abs(beam.pa-pa_out)
        if not(diffmaj<1.0/3600 and diffmin<1.0/3600 and diffpa<0.5):
            print filename,"major {0: 5.3e}deg -> {1: 5.3e}pix -> {2: 5.3e}deg | {3:5.3e} FAIL".format(beam.a,pixbeam1.a,major_out, diffmaj)
            print filename,"minor {0: 5.3e}deg -> {1: 5.3e}pix -> {2: 5.3e}deg | {3:5.3e} FAIL".format(beam.b,pixbeam1.b,minor_out, diffmin)
            print filename,"pa    {0: 5.3e}deg -> {1: 5.3e}deg -> {2: 5.3e}deg | {3:5.3e} FAIL".format(beam.pa,pixbeam1.pa,pa_out, diffpa)
            break


