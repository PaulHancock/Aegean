#! /usr/bin/env python

"""
Implementation of a marching squares algorithm.
With reference to http://devblog.phillipspiess.com/2010/02/23/better-know-an-algorithm-1-marching-squares/
but written in python
by Paul Hancock
May 2014
"""

from copy import copy
import logging
import numpy as np


class MarchingSquares():
    """
    """
    NOWHERE = 0b0000
    UP = 0b0001
    DOWN = 0b0010
    LEFT = 0b0100
    RIGHT = 0b1000


    def __init__(self, data):
        self.prev = self.NOWHERE
        self.next = self.NOWHERE
        self.data = np.nan_to_num(data)
        self.xsize, self.ysize = data.shape
        self.perimeter = self.do_march()
        return

    def dir2str(self,d):
        if d == self.UP:
            return "U"
        elif d == self.DOWN:
            return "D"
        elif d == self.LEFT:
            return "L"
        elif d== self.RIGHT:
            return "R"
        else:
            return "x"

    def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, col in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j
        return None

    def step(self, x, y):
        """
        Move from the current location to the next
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

        #what is the next step?
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
        """
        if not(0 <= x < self.xsize) or not(0 <= y < self.ysize):
            return False
        if self.data[x, y] == 0:
            return False
        if not np.isfinite(self.data[x, y]):
            return False
        return True

    def walk_perimeter(self, startx, starty):
        """
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
            elif self.next == self.NOWHERE:
                break
            else:
                #not sure what to do here
                logging.warn("Failed to determine next step")
                break
            #stop when we return to the starting location
            if x == startx and y == starty:
                break
            #if i>max_tries:
            #	break
            #i+=1
        return points

    def do_march(self):
        """
        March about and trace the outline of our object
        """
        x, y = self.find_start_point()
        perimeter = self.walk_perimeter(x, y)
        return perimeter

    def _blank_within(self, perimeter):
        """
        Blank all the pixels within the given perimeter.
        :param perimeter:
        :return None:
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
            for i in xrange(p[1]+1, self.data.shape[1]):
                q = p[0], i
                # stop when we reach another part of the perimeter
                if q in perimeter:
                    break
                # fill everything in between, even inclusions
                self.data[q] = 0

        return

    def do_march_all(self):
        """
        Recursive march in the case that we have a fragmented shape
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
                


if __name__ == '__main__':
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    test_array = np.random.randint(1, size=(9, 9))
    test_array[0:3, 0:3] = np.ones((3, 3))
    test_array[0, 0] = 0
    # test_array[2:5,2:5]=np.ones((3,3))
    test_array = np.array(test_array, dtype=np.float)
    test_array[np.where(test_array == 0)] = np.nan
    print test_array
    msq = MarchingSquares(test_array)
    print msq.perimeter
    residual = test_array.copy()
    for p in msq.perimeter:
        try:
            residual[p] = 2
        except IndexError:
            pass
    print residual
