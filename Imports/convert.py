#! env python
'''This module contains functions for converting coordinates between epochs
and between different representations. All conversions are done in house so 
as to avoid dependancies on other modules or external code. This will be at
the cost of speed but will give increased portability.
eg
>>> import convert
>>> convert.hms_to_dec('12:34:45.5')
???
>>> convert.ra_to_dec(12,34,45.5)
???

'''

from math import sin,cos,pi, acos,asin,atan, floor

def sgn(x):
	return [-1,1][x>0]

################## Silly helper functions for working in degrees ############
def rad_to_dec(x):
	"""Convert radians to degrees."""
	return x*180/pi

def dec_to_rad(x):
	"""Convert degrees to radians."""
	return x*pi/180

def cosd(x):
	"""Return cosine of x, when x is assumed to be in degrees."""
	return cos(rad_to_dec(x))

def acosd(x):
	"""Return arc-cosine of x in degrees."""
	return rad_to_dec(acos(x))
	
def sind(x):
	"""Return sine of x, when x is assumed to be in radians."""
	return sin(rad_to_dec(x))
	
def asind(x):
	"""Return arcsine of x in degrees."""
	return rad_to_dec(asin(x))

def atand(x):
	"""return arctan of x in degrees."""
	return rad_to_dec(atan(x))

############ More general conversion functions. #############################
def ra2dec(ra):
	"""
	Accepts a string right ascention and converts it to decimal degrees
	requires hh:mm[:ss.s]
	"""
	r = ra.replace(':',' ').split()
	if len(r) == 2:
		r.append(0.0)
	return (float(r[0]) + float(r[1])/60.0 + float(r[2])/3600.0)*15

def dec2dec(dec):
	"""
	Accepts a string declination and converts it to decimal degrees
	requires +/-dd:mm[:ss.s]
	"""
	d = dec.split(':')
	if len(d) == 2:
		d.append(0.0)
	if d[0].startswith('-') or float(d[0]) < 0:
		return float(d[0]) - float(d[1])/60.0 - float(d[2])/3600.0
	return float(d[0]) + float(d[1])/60.0 + float(d[2])/3600.0

def dec2dms(x):
	if x<0:
		sign='-'
	else:
		sign='+'
	x=abs(x)
	d=int(floor(x))
	m=int(floor((x-d)*60))
	s=float(( (x-d)*60-m)*60)
	return '{0}{1:02d}:{2:02d}:{3:05.2f}'.format(sign,d,m,s)

def dec2hms(x):
	x=x/15.0
	h=int(x)
	x=(x-h)*60
	m=int(x)
	s=(x-m)*60
	return	'{0:02d}:{1:02d}:{2:05.2f}'.format(h,m,s)

def eq_to_gal(ra,dec):
	"""Convert equatorial ra,dec to galactic l,b.
	Input is assumed to be in degrees.
	<Copied from TM's code>"""
	# J2000 values from Galactic Astronomy pg. 30-31
	dec_gp = 27.12825
	ra_gp = 192.85948
	l_cp = 122.932  # note error in GA
	b = asind(sind(dec_gp)*sind(dec) + cosd(dec_gp)*cosd(dec)*cosd(ra - ra_gp))
	cosb = cosd(b)
	
	sinl = cosd(dec)*sind(ra - ra_gp) / cosb
	cosl = (cosd(dec_gp)*sind(dec) - sind(dec_gp)*cosd(dec)*cosd(ra - ra_gp)) / cosb
	
	if sinl >= 0 and cosl >= 0:
		l = l_cp - asind(sinl)
	elif sinl >= 0 and cosl < 0:
		l = l_cp - (180 - asind(sinl))
	elif sinl < 0 and cosl < 0:
		l = l_cp - (180 - asind(sinl))
	else:
		l = l_cp - (360 + asind(sinl))
	
	if l < 0:
		l += 360
	return (l, b)


def gal_to_eq(l,b):
	"""Convert galactic l,b to equatorial ra,dec.
	Input is assumed to be in degrees.
	<Currently copied from TM's code. I want a reference though>"""
	c1 = 27.12825
	c2 = 192.85948
	c3 = 33
	dec=cosd(b)*cosd(c1)*sind(l-c3)+sind(b)*sind(c1)
	dec=asind(dec)
	ra=sind(b)*cosd(c1)-cosd(b)*sind(c1)*sind(l-c3)
	ra=cosd(b)*cosd(l-c3)/ra
	ra=atand(ra) + c2
	return (ra,dec)
	
