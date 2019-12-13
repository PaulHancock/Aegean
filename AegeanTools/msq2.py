#! /usr/bin/env python
"""
Provie a class which performs the marching squares algorithm on an image.
The desired output is a set of regions / contours.
"""

from __future__ import print_function

__author__ = "Paul Hancock"

from copy import copy
import numpy as np


class MarchingSquares():
    """
    Implementation of a marching squares algorithm.
    With reference to http://devblog.phillipspiess.com/2010/02/23/better-know-an-algorithm-1-marching-squares/
    but written in python
    """

    NOWHERE = 0b0000
    UP = 0b0001
    DOWN = 0b0010
    LEFT = 0b0100
    RIGHT = 0b1000

    def __init__(self, data):
        self.prev = self.NOWHERE
        self.next = self.NOWHERE
        self.data = np.nan_to_num(data)  # set all the nan values to be zero
        self.xsize, self.ysize = data.shape
        self.perimeter = self.do_march()
        return

    def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, _ in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j

    def step(self, x, y):
        """
        Move from the current location to the next

        Parameters
        ----------
        x, y : int
            The current location
        """
        up_left = self.solid(x - 1, y - 1)
        up_right = self.solid(x, y - 1)
        down_left = self.solid(x - 1, y)
        down_right = self.solid(x, y)

        state = 0
        self.prev = self.next
        # which cells are filled?
        if up_left:
            state |= 1
        if up_right:
            state |= 2
        if down_left:
            state |= 4
        if down_right:
            state |= 8

        # what is the next step?
        if state in [1, 5, 13]:
            self.next = self.UP
        elif state in [2, 3, 7]:
            self.next = self.RIGHT
        elif state in [4, 12, 14]:
            self.next = self.LEFT
        elif state in [8, 10, 11]:
            self.next = self.DOWN
        elif state == 6:
            if self.prev == self.UP:
                self.next = self.LEFT
            else:
                self.next = self.RIGHT
        elif state == 9:
            if self.prev == self.RIGHT:
                self.next = self.UP
            else:
                self.next = self.DOWN
        else:
            self.next = self.NOWHERE
        return

    def solid(self, x, y):
        """
        Determine whether the pixel x,y is nonzero

        Parameters
        ----------
        x, y : int
            The pixel of interest.

        Returns
        -------
        solid : bool
            True if the pixel is not zero.
        """
        if not(0 <= x < self.xsize) or not(0 <= y < self.ysize):
            return False
        if self.data[x, y] == 0:
            return False
        return True

    def walk_perimeter(self, startx, starty):
        """
        Starting at a point on the perimeter of a region, 'walk' the perimeter to return
        to the starting point. Record the path taken.

        Parameters
        ----------
        startx, starty : int
            The starting location. Assumed to be on the perimeter of a region.

        Returns
        -------
        perimeter : list
            A list of pixel coordinates [ [x1,y1], ...] that constitute the perimeter of the region.
        """
        # checks
        startx = max(startx, 0)
        startx = min(startx, self.xsize)
        starty = max(starty, 0)
        starty = min(starty, self.ysize)

        points = []

        x, y = startx, starty

        while True:
            self.step(x, y)
            if 0 <= x <= self.xsize and 0 <= y <= self.ysize:
                points.append((x, y))
            if self.next == self.UP:
                y -= 1
            elif self.next == self.LEFT:
                x -= 1
            elif self.next == self.DOWN:
                y += 1
            elif self.next == self.RIGHT:
                x += 1
            # stop if we meet some kind of error
            elif self.next == self.NOWHERE:
                break
            # stop when we return to the starting location
            if x == startx and y == starty:
                break
        return points

    def do_march(self):
        """
        March about and trace the outline of our object

        Returns
        -------
        perimeter : list
            The pixels on the perimeter of the region [[x1, y1], ...]
        """
        x, y = self.find_start_point()
        perimeter = self.walk_perimeter(x, y)
        return perimeter

    def _blank_within(self, perimeter):
        """
        Blank all the pixels within the given perimeter.

        Parameters
        ----------
        perimeter : list
            The perimeter of the region.

        """
        # Method:
        # scan around the perimeter filling 'up' from each pixel
        # stopping when we reach the other boundary
        for p in perimeter:
            # if we are on the edge of the data then there is nothing to fill
            if p[0] >= self.data.shape[0] or p[1] >= self.data.shape[1]:
                continue
            # if this pixel is blank then don't fill
            if self.data[p] == 0:
                continue

            # blank this pixel
            self.data[p] = 0

            # blank until we reach the other perimeter
            for i in range(p[1]+1, self.data.shape[1]):
                q = p[0], i
                # stop when we reach another part of the perimeter
                if q in perimeter:
                    break
                # fill everything in between, even inclusions
                self.data[q] = 0

        return

    def do_march_all(self):
        """
        Recursive march in the case that we have a fragmented shape.

        Returns
        -------
        perimeters : [perimeter1, ...]
           The perimeters of all the regions in the image.

        See Also
        --------
        :func:`AegeanTools.msq2.MarchingSquares.do_march`
        """
        # copy the data since we are going to be modifying it
        data_copy = copy(self.data)

        # iterate through finding an island, creating a perimeter,
        # and then blanking the island
        perimeters = []
        p = self.find_start_point()
        while p is not None:
            x, y = p
            perim = self.walk_perimeter(x, y)
            perimeters.append(perim)
            self._blank_within(perim)
            p = self.find_start_point()

        # restore the data
        self.data = data_copy
        return perimeters
