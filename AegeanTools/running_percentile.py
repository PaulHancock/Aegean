#! /usr/bin/env python
import bisect
import math
import numpy as np
from blist import *

class RunningPercentiles():
	"""
	Calculate the [0,25,50,75,100]th percentiles of the data
	by updating the sorted list of all our data points
	
	Most useful when the new/old lists of pixels are small
	compared to the total list size.
	"""
	def __init__(self):
		self.slist = blist([])
		self.percentiles = [0,0.25,0.5,0.75,1.0]
		return
	
	def set_percentiles(self,plist):
		"""
		Set the percentile levels that are reported.
		plist =[a,b,c,...] where percentiles are 0->1
		"""
		self.percentiles=plist
		return

	#@profile
	def add(self,dlist):
		"""
		Add a list of elements to our collection, keeping it in order.
		Don't add nan/inf values
		""" 
		#forward sort add
		to_add = sorted([d for d in dlist if np.isfinite(d)])

		if len(to_add)==0:
			return
		if len(self.slist)==0:
			self.slist=blist(to_add)
			return

		add_idx = 0
		add_idx_max = len(to_add)
		add_val = to_add[add_idx]
		idx=0
		slist_len=len(self.slist)
		while True:
			if idx==slist_len:
				break
			if self.slist[idx] >=add_val:
				self.slist.insert(idx,add_val)
				slist_len+=1
				add_idx+=1
				if add_idx==add_idx_max:
					break #we added them all
				add_val=to_add[add_idx]
			idx+=1
		#add the remaining entries to the end of the slist
		if add_idx<len(to_add):
			self.slist.extend(to_add[add_idx:])
		return

	def sub(self,dlist):
		"""
		Remove a list of elements from our collection.
		"""
		#reverse sort remove
		to_remove = sorted([d for d in dlist if np.isfinite(d)])

		if len(to_remove)==0:
			return
		remove_idx = len(to_remove)-1
		#single pass removal
		remove_val = to_remove[remove_idx]
		for idx in xrange(len(self.slist)-1,-1,-1):
			if self.slist[idx]==remove_val:
				del self.slist[idx]
				remove_idx-=1
				remove_val = to_remove[remove_idx]
				if remove_idx<0:
					break #we removed them all
		return

	def score(self):
		"""
		Report the percentile scores of the accumulated data
		"""
		dlen=len(self.slist)
		if dlen<=1:
			return [None for i in self.percentiles]
		vals=[]
		for p in self.percentiles:
			idx = (dlen-1)*p
			frac = idx-math.trunc(idx)
			idx = math.trunc(idx) 
			if frac<0.001: #this avoids problems with the end of the list
				vals.append( self.slist[idx] )
			else:
				vals.append( (1-frac)*self.slist[idx]+frac*self.slist[idx+1] )
		return vals

def test_running_percentiles():
	import numpy as np
	a=np.arange(101)
	correct_answers = [0, 25, 50, 75, 100]
	rp = RunningPercentiles()
	rp.add(a)
	answer = rp.score()
	if answer == correct_answers:
		print "RunningPercentiles: TEST 1 PASS"
	rp.add(np.arange(101,126))
	rp.sub(np.arange(25))
	answer = rp.score()
	correct_answers = [ 25, 50, 75, 100, 125]
	if answer == correct_answers:
		print "RunningPercentiles: TEST 2 PASS"
	return

if __name__ == '__main__':
	test_running_percentiles()
