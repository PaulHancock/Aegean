#! /usr/bin/env python
"""
Cluster and crossmatch tools and analysis functions.

Includes:
- DBSCAN clustering
"""

import logging
import math

import numpy as np
from sklearn.cluster import DBSCAN

from .angle_tools import bear, gcd
from .catalogs import load_table, table_to_source_list
from .wcs_helpers import Beam

__author__ = "Paul Hancock"

log = logging.getLogger('Aegean')

cc2fwhm = (2 * math.sqrt(2 * math.log(2)))
fwhm2cc = 1/cc2fwhm


def norm_dist(src1, src2):
    """
    Calculate the normalised distance between two sources.
    Sources are elliptical Gaussians.

    The normalised distance is calculated as the GCD distance between the
    centers, divided by quadrature sum of the radius of each ellipse along
    a line joining the two ellipses.

    For ellipses that touch at a single point, the normalized distance
    will be 1/sqrt(2).

    Parameters
    ----------
    src1, src2 : object
        The two positions to compare. Objects must have the following
        parameters: (ra, dec, a, b, pa).

    Returns
    -------
    dist: float
        The normalised distance.

    """
    if np.all(src1 == src2):
        return 0
    dist = gcd(src1.ra, src1.dec, src2.ra, src2.dec)  # degrees

    # the angle between the ellipse centers
    phi = bear(src1.ra, src1.dec, src2.ra, src2.dec)  # Degrees
    # Calculate the radius of each ellipse
    # along a line that joins their centers.
    r1 = src1.a*src1.b / np.hypot(src1.a * np.sin(np.radians(phi - src1.pa)),
                                  src1.b * np.cos(np.radians(phi - src1.pa)))
    r2 = src2.a*src2.b / np.hypot(src2.a * np.sin(np.radians(180 + phi - src2.pa)),
                                  src2.b * np.cos(np.radians(180 + phi - src2.pa)))
    R = dist / (np.hypot(r1, r2) / 3600)
    return R


def sky_dist(src1, src2):
    """
    Great circle distance between two sources. A check is made to determine if
    the two sources are the same object, in this case the distance is zero.

    Parameters
    ----------
    src1, src2 : object
        Two sources to check. Objects must have parameters (ra,dec) in degrees.

    Returns
    -------
    distance : float
        The distance between the two sources.

    See Also
    --------
    :func:`AegeanTools.angle_tools.gcd`
    """

    if np.all(src1 == src2):
        return 0
    return gcd(src1.ra, src1.dec, src2.ra, src2.dec)  # degrees


def pairwise_ellpitical_binary(sources, eps, far=None):
    """
    Do a pairwise comparison of all sources and determine if they have a
    normalized distance within eps.

    Form this into a matrix of shape NxN.


    Parameters
    ----------
    sources : list
        A list of sources (objects with parameters: ra,dec,a,b,pa)

    eps : float
        Normalised distance constraint.

    far : float
        If sources have a dec that differs by more than this amount then they
        are considered to be not matched. This is a short-cut around performing
        GCD calculations.

    Returns
    -------
    prob : numpy.ndarray
        A 2d array of True/False.

    See Also
    --------
    :func:`AegeanTools.cluster.norm_dist`
    """
    if far is None:
        far = max(a.a/3600 for a in sources)
    ls = len(sources)
    distances = np.zeros((ls, ls), dtype=bool)
    for i in range(ls):
        for j in range(i, ls):
            if i == j:
                distances[i, j] = False
                continue
            src1 = sources[i]
            src2 = sources[j]
            if src2.dec - src1.dec > far:
                break
            if abs(src2.ra - src1.ra)*np.cos(np.radians(src1.dec)) > far:
                continue
            distances[i, j] = norm_dist(src1, src2) > eps
            distances[j, i] = distances[i, j]
    return distances


def regroup_dbscan(srccat, eps=4):
    """
    Regroup the islands of a catalog according using DBSCAN.

    Return a list of island groups.

    Parameters
    ----------
    srccat : [ `object`]
        A list of objects with parameters ra,dec (both in decimal degrees)
    eps : float
        maximum normalized distance within which sources are considered to be
        grouped

    Returns
    -------
    islands : list of lists
        Each island contains integer indices for members from srccat
        (in descending dec order).

    """
    log.info("Regrouping islands within catalog")

    log.debug("converting ra/dec -> x,y,z")
    # extract ra/dec
    ras = np.radians(np.array([s.ra for s in srccat]))
    decs = np.radians(np.array([s.dec for s in srccat]))

    # convert to cartesian coords
    y = np.cos(decs)
    x = np.cos(ras)*y
    y *= np.sin(ras)
    z = np.sin(decs)

    log.debug("Constructing X array")
    X = np.hstack([x[:, None], y[:, None], z[:, None]])

    log.debug("Clustering")
    # run clustering algorighm
    db = DBSCAN(eps=eps, min_samples=1).fit(X)

    log.debug("Constructing groups")
    # count labels and regroup accordingly
    labels = db.labels_
    unique_labels = set(labels)
    groups = [[]]*len(unique_labels)
    for i, l in enumerate(unique_labels):
        group = list(map(srccat.__getitem__, np.where(labels == l)[0]))
        groups[i] = group

    log.info("Found {0:d} clusters".format(len(unique_labels)))

    log.debug("Labeling/sorting sources")
    islands = []
    # now that we have the groups, we relabel the sources to have
    # (island,component) in flux order note that the order of sources within an
    # island list is not changed - just their labels
    for isle, group in enumerate(groups):
        for comp, src in enumerate(sorted(group,
                                          key=lambda x: -1*x.peak_flux)):
            src.island = isle
            src.source = comp
        islands.append(group)

    sources = []
    for group in islands:
        sources.append(group)
    return sources


def regroup_vectorized(srccat, eps, far=None, dist=norm_dist):
    """
    Regroup the islands of a catalog according to their normalised distance.

    Assumes srccat is recarray-like for efficiency.
    Return a list of island groups.

    Parameters
    ----------
    srccat : np.rec.arry or pd.DataFrame
        Should have the following fields[units]:
        ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]
    eps : float
        maximum normalised distance within which sources are considered to be
        grouped
    far : float
        (degrees) sources that are further than this distance apart will not
        be grouped, and will not be tested.
        Default = 0.5.
    dist : func
        a function that calculates the distance between a source and each
        element of an array of sources.
        Default = :func:`AegeanTools.cluster.norm_dist`

    Returns
    -------
    islands : list of lists
        Each island contians integer indices for members from srccat
        (in descending dec order).
    """
    if far is None:
        far = 0.5  # 10*max(a.a/3600 for a in srccat)

    # most negative declination first
    # XXX: kind='mergesort' ensures stable sorting for determinism.
    #      Do we need this?
    order = np.argsort(srccat.dec, kind='mergesort')[::-1]
    # TODO: is it better to store groups as arrays even if appends are more
    #       costly?
    groups = [[order[0]]]
    for idx in order[1:]:
        rec = srccat[idx]
        # TODO: Find out if groups are big enough for this to give us a speed
        #       gain. If not, get distance to all entries in groups above
        #       decmin simultaneously.
        decmin = rec.dec - far
        for group in reversed(groups):
            # when an island's largest (last) declination is smaller than
            # decmin, we don't need to look at any more islands
            if srccat.dec[group[-1]] < decmin:
                # new group
                groups.append([idx])
            rafar = far / np.cos(np.radians(rec.dec))
            group_recs = np.take(srccat, group, mode='clip')
            group_recs = group_recs[abs(rec.ra - group_recs.ra) <= rafar]
            if len(group_recs) and dist(rec, group_recs).min() < eps:
                group.append(idx)
                break
        else:
            # new group
            groups.append([idx])

    # TODO?: a more numpy-like interface would return only an array providing
    #        the mapping:
    #    group_idx = np.empty(len(srccat), dtype=int)
    #    for i, group in enumerate(groups):
    #        group_idx[group] = i
    #    return group_idx
    return groups


def regroup(catalog, eps, far=None, dist=norm_dist):
    """
    Regroup the islands of a catalog according to their normalised distance.
    Return a list of island groups. Sources have their (island,source)
    parameters relabeled.


    Parameters
    ----------
    catalog : str or object
        Either a filename to read into a source list, or a list of objects with
        the following properties[units]: ra[deg], dec[deg], a[arcsec],
        b[arcsec],pa[deg], peak_flux[any]

    eps : float
        maximum normalised distance within which sources are considered to be
        grouped

    far : float
        (degrees) sources that are further than this distance appart will not
        be grouped, and will not be tested. Default = None.

    dist : func
        a function that calculates the distance between two sources must accept
        two SimpleSource objects. Default =
        :func:`AegeanTools.cluster.norm_dist`

    Returns
    -------
    islands : list
        A list of islands. Each island is a list of sources.

    See Also
    --------
    :func:`AegeanTools.cluster.norm_dist`
    """

    if isinstance(catalog, str):
        table = load_table(catalog)
        srccat = table_to_source_list(table)
    else:
        try:
            srccat = catalog
            _ = catalog[0].ra, catalog[0].dec, catalog[0].a, catalog[0].b
            _ = catalog[0].pa, catalog[0].peak_flux

        except AttributeError as e:
            log.error("catalog is not understood.")
            log.error("catalog: Should be a list of objects with the " +
                      "following properties[units]:\n" +
                      "ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg]," +
                      " peak_flux[any]")
            raise e

    log.info("Regrouping islands within catalog")
    log.debug("Calculating distances")

    if far is None:
        far = 0.5  # 10*max(a.a/3600 for a in srccat)

    srccat_array = np.rec.fromrecords(
        [(s.ra, s.dec, s.a, s.b, s.pa, s.peak_flux)
         for s in srccat],
        names=['ra', 'dec', 'a', 'b', 'pa', 'peak_flux'])
    groups = regroup_vectorized(srccat_array, eps=eps, far=far, dist=dist)
    groups = [[srccat[idx] for idx in group]
              for group in groups]

    islands = []
    # now that we have the groups, we relabel the sources to have
    # (island,component) in flux order note that the order of sources within an
    # island list is not changed - just their labels
    for isle, group in enumerate(groups):
        for comp, src in enumerate(sorted(group,
                                          key=lambda x: -1*x.peak_flux)):
            src.island = isle
            src.source = comp
        islands.append(group)

    sources = []
    for group in islands:
        sources.append(group)
    return sources


def resize(catalog, ratio=None, psfhelper=None):
    """
    Resize all the sources in a given catalogue. Either use a ratio to blindly
    scale all sources by the same amount, or use a psf map to deconvolve the
    sources and then convolve them with the new psf

    Sources that cannot be rescaled are not returned

    Parameters
    ----------
    catalog : list
        List of objects

    ratio : float, default=None
        Ratio for scaling the sources

    psfhelper : :py:class:`AegeanTools.wcs_helpers.WCSHelper`, default=None
        A wcs helper object that contains psf information for the target
        image/projection

    Returns
    -------
    catalog : list
        Modified list of objects
    """

    src_mask = np.ones(len(catalog), dtype=bool)

    # check to see if the input catalog contains psf information
    has_psf = getattr(catalog[0], "psf_a", None) is not None

    # If ratio is provided we just the psf by this amount
    if ratio is not None:
        log.info(
            "Using ratio of {0} to scale input source shapes".format(ratio))

        for i, src in enumerate(catalog):
            # the new source size is the previous size, convolved with the
            # expanded psf
            src.a = np.sqrt(
                src.a ** 2 + (src.psf_a) ** 2 * (1 - 1 / ratio ** 2)
            )
            src.b = np.sqrt(
                src.b ** 2 + (src.psf_b) ** 2 * (1 - 1 / ratio ** 2)
            )
            # source with funky a/b are also rejected
            if not np.all(np.isfinite((src.a, src.b))):
                log.info(
                    ("Excluding source ({0.island},{0.source})" +
                     " due to bad psf ({0.a},{0.b},{0.pa})").format(src))
                src_mask[i] = False

    # if we know the psf from the input catalogue (has_psf), or if it was
    # provided via a psf map then we use that psf.
    elif psfhelper is not None or has_psf:
        for i, src in enumerate(catalog):
            if (src.psf_a <= 0) or (src.psf_b <= 0):
                src_mask[i] = False
                log.info(
                    ("Excluding source ({0.island},{0.source})" +
                     "due to psf_a/b <=0").format(src)
                )
                continue
            if has_psf:
                catbeam = Beam(src.psf_a / 3600, src.psf_b / 3600, src.psf_pa)
            else:
                catbeam = Beam(*psfhelper.get_psf_sky2sky(src.ra, src.dec))
            imbeam = psfhelper.get_skybeam(src.ra, src.dec)
            # If either of the above are None then we skip this source.
            if catbeam is None or imbeam is None:
                unknown = []
                if catbeam is None:
                    unknown.append("input catalogue")
                if imbeam is None:
                    unknown.append("image")
                src_mask[i] = False
                log.info(
                    ("Excluding source ({0.island},{0.source}) due to " +
                     "lack of psf knowledge in {1}").format(src,
                                                            ",".join(unknown))
                    )
                continue

            # TODO: The following assumes that the various psf's are scaled
            # versions of each other and makes no account for differing
            # position angles. This needs to be checked and/or addressed.

            # deconvolve the source shape from the catalogue psf
            src.a = (src.a / 3600) ** 2 - catbeam.a ** 2 + \
                imbeam.a ** 2  # degrees

            # clip the minimum source shape to be the image psf
            if src.a < 0:
                src.a = imbeam.a * 3600  # arcsec
            else:
                src.a = np.sqrt(src.a) * 3600  # arcsec

            src.b = (src.b / 3600) ** 2 - catbeam.b ** 2 + imbeam.b ** 2
            if src.b < 0:
                src.b = imbeam.b * 3600  # arcsec
            else:
                src.b = np.sqrt(src.b) * 3600  # arcsec
    else:
        log.info("Not scaling input source sizes")
    # return only the sources where resizing was possible
    out_cat = list(map(catalog.__getitem__, np.where(src_mask)[0]))
    return out_cat


def check_attributes_for_regroup(catalog):
    """
    Check that the catalog has all the attributes reqired for the regrouping
    task.

    Parameters
    ----------
    catalog : list
        List of python objects, ideally derived from
        :py:class:`AegeanTools.models.SimpleSource`

    Returns
    -------
    result : bool
        True if the first entry in the catalog has the required attributes
    """
    src = catalog[0]
    missing = []
    for att in ['ra', 'dec', 'a', 'b', 'pa']:
        if not hasattr(src, att):
            missing.append(att)

    if missing:
        log.error("catalog is not understood.")
        log.error(
            "catalog: Should be a list of objects with the following properties[units]:")
        log.error("ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg]")
        return False
    return True
