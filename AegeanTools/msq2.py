#! /usr/bin/evn python

"""
Implementation of a marching square algorithm.
With reference to http://devblog.phillipspiess.com/2010/02/23/better-know-an-algorithm-1-marching-squares/
but written in python
by Paul Hancock
May 2014
"""

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
        self.perimeter = self.doMarch()
        return

    def findStartPoint(self):
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
        upLeft = self.solid(x - 1, y - 1)
        upRight = self.solid(x, y - 1)
        downLeft = self.solid(x - 1, y)
        downRight = self.solid(x, y)

        state = 0
        self.prev = self.next
        # which cells are filled?
        if upLeft:
            state |= 1
        if upRight:
            state |= 2
        if downLeft:
            state |= 4
        if downRight:
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
        if x < 0 or y < 0 or x >= self.xsize or y >= self.ysize:
            return False
        if self.data[x, y] == 0:
            return False
        if not np.isfinite(self.data[x, y]):
            return False
        return True

    def walkPerimeter(self, startx, starty):
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

    def doMarch(self):
        """
        March about and trace the outline of our object
        """

        perimeter = []
        p = self.findStartPoint()
        while p is not None:
            x, y = self.findStartPoint()
            perim = self.walkPerimeter(x, y)
            perimeter.extend(perim)
            p = None
            # For fragmented islands [Doesn't work]
            # x,y = zip(*[p for p in perim if p[0]<self.xsize and p[1] < self.ysize])
            # self.data[x,y] = 0
            # p = self.findStartPoint()
        return perimeter


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
        except:
            pass
    print residual
