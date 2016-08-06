#! /usr/bin/env python
import bisect
import math
import numpy as np
from blist import blist


class RunningPercentiles():
    """
    Calculate the [0,25,50,75,100]th percentiles of the data
    by updating the sorted list of all our data points

    Most useful when the new/old lists of pixels are small
    compared to the total list size.
    """

    def __init__(self):
        self.slist = blist([])
        self.percentiles = [0, 0.25, 0.5, 0.75, 1.0]
        return

    def set_percentiles(self, plist):
        """
        Set the percentile levels that are reported.
        plist =[a,b,c,...] where percentiles are 0->1
        """
        self.percentiles = plist
        return

    def add(self, dlist):
        """
        Add a list of elements to our collection, keeping it in order.
        Don't add nan/inf values
        """

        to_add = blist(d for d in dlist if np.isfinite(d))

        if len(to_add) == 0:
            return
        if len(self.slist) == 0:
            self.slist = blist(sorted(to_add))
            return

        #cheap version (works better for large length arrays)
        # ie when we have grid*box >1e6
        # 2.13795 s for test image
        #self.slist=blist(merge(to_add,self.slist))

        #cheaper!
        #1.96215 s for test image
        self.slist.extend(to_add)
        self.slist.sort()
        return

    def sub(self, dlist):
        """
        Remove a list of elements from our collection.
        """

        # reverse sort remove
        to_remove = sorted(d for d in dlist if np.isfinite(d))

        #orig version
        # 2.6s
        # if len(to_remove) == 0:
        #     return
        # remove_idx = len(to_remove) - 1
        # #single pass removal
        # remove_val = to_remove[remove_idx]
        # for idx in xrange(len(self.slist) - 1, -1, -1):
        #     if self.slist[idx] == remove_val:
        #         del self.slist[idx]
        #         remove_idx -= 1
        #         remove_val = to_remove[remove_idx]
        #         if remove_idx < 0:
        #             break  #we removed them all

        #bisect
        for val in to_remove:
            del self.slist[bisect.bisect(self.slist,val)-1]

        return

    def score(self):
        """
        Report the percentile scores of the accumulated data
        """
        dlen = len(self.slist)
        if dlen <= 1:
            return [None for i in self.percentiles]
        vals = []
        for p in self.percentiles:
            idx = (dlen - 1) * p
            frac = idx - math.trunc(idx)
            idx = math.trunc(idx)
            if frac < 0.001:  # this avoids problems with the end of the list
                vals.append(self.slist[idx])
            else:
                vals.append((1 - frac) * self.slist[idx] + frac * self.slist[idx + 1])
        return vals

def test_running_percentiles():
    """
    Test for RunningPercentiles class, with a super basic list input
    :return:
    """
    import numpy as np

    a = np.arange(11)
    correct_answers = [0, 2.5, 5.0, 7.5, 10]
    rp = RunningPercentiles()
    rp.add(a)
    print rp.slist
    answer = rp.score()
    if answer == correct_answers:
        print "RunningPercentiles: TEST 1 PASS"
    to_add = np.arange(11, 26)
    np.random.shuffle(to_add)
    rp.add(to_add)
    to_sub = np.arange(15)
    np.random.shuffle(to_sub)
    rp.sub(to_sub)
    print rp.slist
    answer = rp.score()
    correct_answers = [15, 17.5, 20, 22.5, 25]
    if answer == correct_answers:
        print "RunningPercentiles: TEST 2 PASS"
    else:
        print "TEST 2 Fail:"
        print "correct=",correct_answers
        print "answer=",answer
    return


if __name__ == '__main__':
    test_running_percentiles()
