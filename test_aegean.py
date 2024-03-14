import logging
from AegeanTools.source_finder import SourceFinder

# configure logging
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)
sf = SourceFinder(log=log)
found = sf.find_sources_in_image('/Users/ahmedhamadto/Documents/Curtin University Australia/Study Material/Year 2/Semester 1/Computer Project/Aegean_testing_cutouts_170-200MHz/193-200MHz_cutout.fits', 
                                rmsin='/Users/ahmedhamadto/Documents/Curtin University Australia/Study Material/Year 2/Semester 1/Computer Project/Aegean_testing_cutouts_170-200MHz/193-200MHz_cutout_rms.fits', 
                                bkgin='/Users/ahmedhamadto/Documents/Curtin University Australia/Study Material/Year 2/Semester 1/Computer Project/Aegean_testing_cutouts_170-200MHz/193-200MHz_cutout_bkg.fits')