#!/usr/bin/env python
# coding=utf-8

# Author    :   Shaoguang Guo && Yang Lu
# Email     :   sgguo@shao.ac.cn
# Institute :   Shanghai Astronomical Observatory

'''
Module to fix the BEAM info
Will read beam info from HISTORY
'''

__author__ = "Guo Shaoguang"
__version__ = 'v1.0_alpha'
__date__ = '2016-07-30'
__institute__ = 'Shanghai Astronomical Observatory'

from astropy.io import fits
import logging
from optparse import OptionParser

def load_file_or_hdu(filename):
    '''
    Load a file from disk and return an HDUList
    If filename is already an HDUList return that instead

    :param filename: filename or HDUList
    :return: HDUList
    '''
    if isinstance(filename,fits.HDUList):
        hdulist = filename
    else:
        try:
            hdulist = fits.open(filename)
        except IOError,e:
            if 'END' in e.message:
                logging.warn(e.message)
                logging.warn('Trying to ignore this,but you should really fix it')
                hdulist = fits.open(filename,ignore_missing_end = True)
            else:
                raise e
    return hdulist

def search_beam(hdulist):
    '''
    Will search the beam infor from the HISTORY
    :param hdulist:
    :return:
    '''
    header = hdulist[0].header
    data = hdulist[0].data
    #print header['HISTORY']
    history = header['HISTORY']
    history_str = str(history)

    #AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00
    if 'BMAJ' in history_str:
        #if 'AIPS' in history_str:
        print 'Yeah, Found the BEAM informations'
        return True
    else:
        print 'Sorry, do not found the BEAM informations'
        return False

def get_beam(hdulist):
    '''
    Will get the beam info
    :param hdulist:
    :return: BMAJ,BMIN,BPA
    '''

    header = hdulist[0].header
    data = hdulist[0].data
    #print header['HISTORY']
    history = header['HISTORY']
    history_str = str(history)
    #print history_str
    #AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00
    print len(history_str)
    print history_str.find('BMAJ')
    loc = history_str.find('BMAJ')
    a = []
    for i in range(47):
        #print history_str[loc+i],
        a.append(history_str[loc+i])
    #print a
    info = str(''.join(a))
    print info
    rst = info.split(' ')
    print rst
    return float(rst[2]),float(rst[5]),float(rst[9])

def set_beam(hdulist,BMAJ,BMIN,BPA,filename):
    '''
    set the beam info to hdulist
    :param hdulist:
    :param BMAJ:
    :param BMIN:
    :param BPA:
    :param filename:the outfile
    :return:
    '''
    header = hdulist[0].header
    header['BMAJ'] = BMAJ
    header['BMIN'] = BMIN
    header['BPA']  = BPA
    header['HISTORY'] = 'Add beam info by {0}'.format(__institute__)

    hdulist.writeto(filename)

if __name__ == '__main__':
    usage = "%prog [--infile] in.fits [--outfile] out.fits"

    parser = OptionParser(usage=usage)

    parser.add_option('-i','--infile',dest='infile',default=None,
                      help='The input fits file')
    parser.add_option('-o','--outfile',dest='outfile',default=None,
                      help='The outfile fits file')
    (options,args) = parser.parse_args()

    infile = ''
    outfile = ''

    if options.infile is None:
        print 'Please specific the input fits file'
    else:
        infile = options.infile
    if options.outfile is None:
        print 'The outfile in default name : out.fits'
        outfile = 'out.fits'
    else:
        outfile = options.outfile

    print infile
    print outfile

    hdulist = load_file_or_hdu(infile)

    found = search_beam(hdulist)

    if found:
        BMAJ,BMIN,BPA = get_beam(hdulist)
        set_beam(hdulist,BMAJ,BMIN,BPA,outfile)
        print 'Had writen the beam informations'
    else:
        print 'Not found the beam informations'
