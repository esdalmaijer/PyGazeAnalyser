# PyeNalysis

__author__ = "Edwin Dalmaijer"


import copy
import numpy
from scipy.interpolate import interp1d

# DEBUG #
#from matplotlib import pyplot
# # # # #


def interpolate_blink(signal, mode='auto', velthresh=5, maxdur=500, margin=10, invalid=-1, edfonly=False):
	
	"""Returns signal with interpolated results, based on a cubic or linear
	interpolation of all blinks detected in the signal; based on:
	https://github.com/smathot/exparser/blob/master/exparser/TraceKit.py
	
	arguments
	signal	--	a vector (i.e. a NumPy array) containing a single
				trace of your signal; alternatively a trial gaze data
				dict as is returned by edfreader can be passed; in this
				case the blink ending events will be used to find blinks
				before the pupil size velocity algorithm will be used
				(NOTE: this means both will be used successively!)
	
	keyword arguments
	mode		--	string indicating what kind of interpolation to use:
				'linear' for a linear interpolation
				'cubic' for a cubic interpolation
				'auto' for a cubic interpolation is possible (i.e.
					when more than four data points are available)
					and linear when this is not the case
				(default = 'auto')
	velthresh	--	pupil size change velocity threshold in arbitrary
				units per sample (default = 5)
	maxdur	--	maximal duration of the blink in samples
				(default = 500)
	margin	--	margin (in samples) to compensate for blink duration
				underestimatiom; blink is extended for detected start
				minus margin, and detected end plus margin
				(default = 10)
	edfonly	--	Boolean indicating whether blinks should ONLY be
				detected using the EDF logs and NOT algorithmically
	
	returns
	signal	--	a NumPy array containing the interpolated signal
	"""
	
	# # # # #
	# input errors
	
	# wrong interpolation method	
	if mode not in ['auto','linear','cubic']:
		raise Exception("Error in pyenalysis.interpolate_missing: mode '%s' is not supported, please use one of the following: 'auto','linear','cubic'" % mode)
	# wrong signal dimension
	if type(signal) != dict:
		if signal.ndim != 1:
			raise Exception("Error in pyenalysis.interpolate_missing: input is not a single signal trace, but has %d dimensions; please provide a 1-dimension array" % signal.ndim)

	# # # # #
	# find blinks
	
	# empty lists, to store blink starts and endings
	starts = []
	ends = []
	
	# edfreader data
	if type(signal) == dict:

		# loop through blinks
		for st, et, dur in signal['events']['Eblk']: # Eblk - list of lists, each containing [starttime, endtime, duration]
			
			# edf time to sample number
			st = numpy.where(signal['edftime']==st)[0]
			et = numpy.where(signal['edftime']==et)[0]
			# if the starting or ending time did not appear in the trial,
			# correct the blink starting or ending point to the first or
			# last sample, respectively
			if len(st) == 0:
				st = 0
			else:
				st = st[0]
			if len(et) == 0:
				et = len(signal['edftime'])
			else:
				et = et[0]
			# compensate for underestimation of blink duration
			if st-margin >= 0:
				st -= margin
			if et+margin < len(signal['size']):
				et += margin
			# do not except blinks that exceed maximal blink duration
			if et-st <= maxdur:
				# append start time and ending time
				starts.append(st)
				ends.append(et)
		# extract pupil size data from signal
		signal = signal['size']
	
	if not edfonly:
		# signal in NumPy array
		# create a velocity profile of the signal
		vprof = signal[1:]-signal[:-1]
		
		# start detection
		ifrom = 0
		while True:
			# blink onset is detected when pupil size change velocity exceeds
			# threshold
			l = numpy.where(vprof[ifrom:] < -velthresh)[0]
			# break when no blink start is detected
			if len(l) == 0:
				break
			# blink start index
			istart = l[0]+ifrom
			if ifrom == istart:
				break
			# reversal (opening of the eye) is detected when pupil size
			# starts to increase with a super-threshold velocity
			l = numpy.where(vprof[istart:] > velthresh)[0]
			# if no reversal is detected, start detection process at istart
			# next run
			if len(l) == 0:
				ifrom = istart
				# reloop
				continue
			# index number of somewhat halfway blink process
			imid = l[0] + istart
			# a blink ending is detected when pupil size increase velocity
			# falls back to zero
			l = numpy.where(vprof[imid:] < 0)[0]
			# if no ending is detected, start detection process from imid
			# next run
			if len(l) == 0:
				ifrom = imid
				# reloop
				continue
			# blink end index
			iend = l[0]+imid
			# start detection process from current blink ending next run
			ifrom = iend
			# compensate for underestimation of blink duration
			if istart-margin >= 0:
				istart -= margin
			if iend+margin < len(signal):
				iend += margin
			# do not except blinks that exceed maximal blink duration
			if iend-istart > maxdur:
				# reloop
				continue
			# if all is well, we append start and ending to their respective
			# lists
			starts.append(istart)
			ends.append(iend)

#	# DEBUG #
#	pyplot.figure()
#	pyplot.title("" % ())
#	pyplot.plot(signal,'ko')
#	pyplot.plot(vprof,'b')
#	# # # # #

	# # # # #
	# interpolate

	# loop through all starting and ending positions
	for i in range(len(starts)):
		
		# empty list to store data points for interpolation
		pl = []
		
		# duration in samples
		duration = ends[i]-starts[i]

		# starting point
		if starts[i] - duration >= 0:
			pl.extend([starts[i]-duration])
		# central points (data between these points will be replaced)
		pl.extend([starts[i],ends[i]])
		# ending point
		if ends[i] + duration < len(signal):
			pl.extend([ends[i]+duration])
		
		# choose interpolation type
		if mode == 'auto':
			# if our range is wide enough, we can interpolate cubicly
			if len(pl) >= 4:
				kind = 'cubic'
			# if not, we use a linear interpolation
			else:
				kind = 'linear'
		else:
			kind = mode[:]
		
		# select values for interpolation function
		x = numpy.array(pl)
		y = signal[x]
		
		# replace any invalid values with trial average
		y[y==invalid] = numpy.mean(signal[signal!=invalid])
		
		# create interpolation function
		intfunc = interp1d(x,y,kind=kind)
		
		# do interpolation
		xint = numpy.arange(starts[i],ends[i])
		yint = intfunc(xint)
		
		# insert interpolated values into signal
		signal[xint] = yint
	
#		# DEBUG #
#		y = numpy.zeros(len(pl)) + max(signal)
#		pyplot.plot(pl,y,'ro')
#	pyplot.plot(signal,'r')
#	# # # # #
		
	return signal


def interpolate_missing(signal, mode='auto', mindur=5, margin=10, invalid=-1):
	
	"""Returns signal with interpolated results, based on a cubic or linear
	interpolation of the invalid data in the signal
	
	arguments
	signal	--	a vector (i.e. a NumPy array) containing a single
				trace of your signal
	
	keyword arguments
	mode		--	string indicating what kind of interpolation to use:
				'linear' for a linear interpolation
				'cubic' for a cubic interpolation
				'auto' for a cubic interpolation is possible (i.e.
					when more than four data points are available)
					and linear when this is not the case
				(default = 'auto')
	mindur	--	minimal amount of consecutive samples to interpolate
				cubically; otherwise a linear interpolation is used;
				this is to prevent weird results in the interpolation
				of very short strings of missing data (default = 5)
	margin	--	margin (in samples) to compensate for missing duration
				underestimatiom; missing is extended for detected start
				minus margin, and detected end plus margin; this helps
				in reducing errors in blink interpolation that has not
				been done by interpolate_blink (default = 10)
	invalid	--	a single value coding for invalid data, e.g. -1 or 0.0
				(default = -1)
	
	returns
	signal	--	a NumPy array containing the interpolated signal
	"""
	
	# # # # #
	# input errors
	
	# wrong interpolation method	
	if mode not in ['auto','linear','cubic']:
		raise Exception("Error in pyenalysis.interpolate_missing: mode '%s' is not supported, please use one of the following: 'auto','linear','cubic'" % mode)
	# wrong signal dimension
	if signal.ndim != 1:
		raise Exception("Error in pyenalysis.interpolate_missing: input is not a single signal trace, but has %d dimensions; please provide a 1-dimension array" % signal.ndim)

	# # # # #
	# find successive strings of missing data
	
	# empty lists for starting and ending indexes
	starts = []
	ends = []
	
	# check if beginning sample is missing, and add to starting indexes if
	# needed (algorithm does not pick up changes before the start or after
	# the end if the signal)
	if signal[0] == invalid:
		starts.append(0)
		si = 1
	else:
		si = 0
	
	# find invalid data
	inval = signal == invalid
	
	# code connected strings of missing data 1
	# (by substracting the previous number from the current, for every
	# missing data index number: this will produce a value of 1 for
	# successive index numbers, and higher values for nonsuccessive ones)
	diff = numpy.diff(inval)
	
	# find out what the index numbers of changes are
	# (i.e.: every difference that is 1)
	changes = numpy.where(diff==True)[0]
	
	# loop through changes, finding start and begining index numbers for
	# strings of successive missings
	for i in range(si,len(changes),2):
		ns = changes[i]-margin
		if ns < 0:
			ns = 0
		starts.append(ns)
	for i in range(1-si,len(changes),2):
		ne = changes[i]+1+margin
		if ne >= len(signal):
			ne = len(signal)-1
		ends.append(ne)
	# if the signal ended on an invalid sample, add the ending index number
	if signal[-1] == invalid:
		ends.append(len(signal)-1)
	
	# # # # #
	# interpolate
	
	# correct start and end point if these are invalid, by replacing them
	# with the trial average
	if signal[0] == invalid:
		signal[0] = numpy.mean(signal[signal != invalid])
	if signal[-1] == invalid:
		signal[-1] = numpy.mean(signal[signal != invalid])
	

	# loop through all starting and ending positions
	for i in range(len(starts)):
		
		# empty list to store data points for interpolation
		pl = []
		
		# duration in samples
		duration = ends[i]-starts[i]
		
		# starting point
		if starts[i] - duration >= 0 and signal[starts[i]-duration] != invalid:
			pl.extend([starts[i]-duration])
		# central points (data between these points will be replaced)
		pl.extend([starts[i],ends[i]])
		# ending point
		if ends[i] + duration < len(signal) and signal[ends[i]+duration] != invalid:
			pl.extend([ends[i]+duration])
		
		# if the duration is too low, use linear interpolation
		if duration < mindur:
			kind = 'linear'
		# if the duration is long enough, choose interpolation type
		else:
			if mode == 'auto':
				# if our range is wide enough, we can interpolate cubicly
				if len(pl) >= 4:
					kind = 'cubic'
				# if not, we use a linear interpolation
				else:
					kind = 'linear'
			else:
				kind = mode[:]
		
		# create interpolation function
		x = numpy.array(pl)
		y = signal[x]
		intfunc = interp1d(x,y,kind=kind)
		
		# do interpolation
		xint = numpy.arange(starts[i],ends[i])
		yint = intfunc(xint)
		
		# insert interpolated values into signal
		signal[xint] = yint
		
	return signal
	

def remove_outliers(signal, maxdev=2.5, invalid=-1, interpolate=True, mode='auto', allowp=0.1):
	
	"""Replaces every outlier with a missing value, then interpolates
	missing values using pyenalysis.interpolate_missing
	
	arguments
	signal	--	a vector (i.e. a NumPy array) containing a single
				trace of your signal
	
	keyword arguments
	maxdev	--	maximal distance between a single sample and the
				signal average in standard deviations (default = 2.5)
	invalid	--	a single value coding for invalid data, e.g. -1 or 0.0;
				outliers will be replaced by this value (default = -1)
	interpolate	--	Boolean indicating whether outliers should be
				should be interpolated (True) or replaced by missing
				values (False) (default = True)
	mode		--	string indicating what kind of interpolation to use:
				'linear' for a linear interpolation
				'cubic' for a cubic interpolation
				'auto' for a cubic interpolation is possible (i.e.
					when more than four data points are available)
					and linear when this is not the case
				(default = 'auto')
	allowp	--	is the standard deviation is below this proportion of
				the mean, outliers will not be removed; this is to
				prevent erroneous removal of outliers in a very steady
				signal (default = 0.1)
	
	returns
	signal	--	signal with outliers replaced by missing or
				interpolated (depending on interpolate keyword
				argument)
	"""
	
	# # # # #
	# input errors
	
	# wrong interpolation method	
	if mode not in ['auto','linear','cubic']:
		raise Exception("Error in pyenalysis.interpolate_missing: mode '%s' is not supported, please use one of the following: 'auto','linear','cubic'" % mode)
	# wrong signal dimension
	if signal.ndim != 1:
		raise Exception("Error in pyenalysis.interpolate_missing: input is not a single signal trace, but has %d dimensions; please provide a 1-dimension array" % signal.ndim)
	
	# # # # #
	# outlier removal
	
	# calculate signal average and standard deviation
	mean = numpy.mean(signal)
	sd = numpy.std(signal)
	
	# stop if SD is too low
	if sd < mean*allowp:
		return signal
	
	# calculate bounds
	lower = mean - maxdev*sd
	upper = mean + maxdev*sd
	
	# find outliers
	outlier = (signal > upper) | (signal < lower)
	
	# replace outliers by invalid code
	signal[outlier] = invalid
	
	# interpolate
	if interpolate:
		signal = interpolate_missing(signal, mode=mode, invalid=invalid)
	
	return signal


def hampel(signal, winlen=12, T=3, focus='centre'):
	
	"""Performs a Hampel filtering, a median based outlier rejection in which
	outliers are detected based on a local median, and are replaced by that
	median (local median is determined in a moving window)
	
	arguments
	
	signal	--	a vector (i.e. a NumPy array) containing a single
				trace of your signal

	keyword arguments

	winlen	--	integer indicating window length (default = 12)
	
	T		--	floating point or integer indicating the maximal
				distance from the surrounding signal that defines
				outliers; distance is measured in a standard deviation
				like measure (S0), based on the local median; a T of 3
				means that any point outside of the range -3*S0 to 3*S0
				is considered an outlier (default = 3)
	focus		--	string indicating where the focus (i.e. the point that
				is being corrected) of the window should be; one of:
					'centre' (window = winlen/2 + i + winlen/2)
					'left' '(window = i + winlen)
					'right' (window = winlen + i)
	"""
	
	if focus == 'centre':
		# half a window length
		hampwinlen = winlen/2
		for i in range(hampwinlen, len(signal)-hampwinlen+1):
			# median for this window
			med = numpy.median(signal[i-hampwinlen:i+hampwinlen])
			# check S0 (standard deviation like measure)
			s0 = 1.4826 * numpy.median(numpy.abs(signal[i-hampwinlen:i+hampwinlen] - med))
			# check outliers
			if signal[i] > T*s0 or signal[i] < T*s0:
				# replace outliers by median
				signal[i] = med

	# if the focus is not the centre
	else:
		# determine the starting position
		if focus == 'left':
			start = 0
			stop = len(signal) - winlen
		elif focus == 'right':
			start = winlen
			stop = len(signal)
		else:
			start = winlen/2
			stop = len(signal) - winlen/2 + 1
		# loop through samples
		for i in range(start, stop):
			# determine window start and stop
			if focus == 'left':
				wstart = i
				wstop = i + winlen
			elif focus == 'right':
				wstart = i - winlen
				wstop = i
			else:
				wstart = i - winlen/2
				wstop = i + winlen/2
			# median for this window
			med = numpy.median(signal[wstart:wstop])
			# check S0 (standard deviation like measure)
			s0 = 1.4826 * numpy.median(numpy.abs(signal[wstart:wstop] - med))
			# check outliers
			if signal[i] > T*s0 or signal[i] < T*s0:
				# replace outliers by median
				signal[i] = copy.deepcopy(med)
	
	return signal


def smooth(signal, winlen=11, window='hanning', lencorrect=True):
	
	"""Smooth a trace, based on: http://wiki.scipy.org/Cookbook/SignalSmooth
	
	arguments
	signal	--	a vector (i.e. a NumPy array) containing a single
				trace of your signal
	
	keyword arguments
	winlen	--	integer indicating window length (default = 11)
	window	--	smoothing type, should be one of the following:
				'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'
				(default = 'hanning')
	lencorrect	--	Boolean indicating if the output (the smoothed signal)
				should have the same length as the input (the raw
				signal); this is not necessarily so because of data
				convolution (default = True)
	
	returns
	signal	--	smoothed signal
	"""
	
	# # # # #
	# input errors
	
	# really small window
	if winlen < 3:
		return signal
	# non-integer window length
	if type(winlen) != int:
		try:
			winlen = int(winlen)
		except:
			raise Exception("Error in pyenalysis.smooth: provided window length ('%s') is not compatible; please provide an integer window length" % winlen)
	# wrong type of window
	if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise Exception("Error in pyenalysis.smooth: windowtype '%s' is not supported; please use one of the following: 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'" % window)
	# wrong signal dimension
	if signal.ndim != 1:
		raise Exception("Error in pyenalysis.smooth: input is not a single signal trace, but has %d dimensions; please provide a 1-dimension array" % signal.ndim)
	# too small a trace
	if signal.size < winlen:
		raise Exception("Error in pyenalysis.smooth: input signal has too few datapoints (%d) for provided window length (%d)" % (signal.size,winlen))
	
	# # # # #
	# smoothing

	# slice to concatenation
	s = numpy.r_[signal[winlen-1:0:-1],signal,signal[-1:-winlen:-1]]

	# this is equivalent to:
	# p1 = signal[winlen-1:0:-1].tolist() # first part of signal reversed
	# p2 = signal.tolist()
	# p3 = signal[-1:-winlen:-1].tolist() # last part of signal reversed
	# s = p1 + p2 + p3

	
	# moving average
	if window == 'flat':
		w = numpy.ones(winlen, 'd')
	# bit more sophisticated smoothing algorithms
	else:
		w = eval("numpy.%s(%d)" % (window,winlen))
	
	# convolve signal, according to chosen smoothing type
	smoothed = numpy.convolve(w/w.sum(), s, mode='valid')
	
	# correct length if necessary
	if lencorrect:
		smoothed = smoothed[(winlen/2-1):(-winlen/2)]
		try:
			smoothed = smoothed[:len(signal)]
		except:
			raise Exception("Error in pyenalysis.smooth: output array is too short (len(output)=%d, len(signal)=%d)" % (len(smoothed),len(signal)))

	return smoothed


# DEBUG #
if __name__ == '__main__':
	from matplotlib import pyplot
	# constants
	N = 200
	INVAL = -1
	# create random data
	a = numpy.random.random_sample(N)
	# manipulate radom data to look like somewhat realictic data
	a = 10 + a*5
	# introduce some missing values
	a[0:10] = INVAL
	a[50:66] = INVAL
	a[150:190] = INVAL
	a[-1] = INVAL
	# introduce ouliers
	for i in [15,16,18,40,197]:
		a[i] = a[i] + numpy.random.random()*30
	# plot the raw data
	pyplot.figure()
	pyplot.plot(a,'ko', label='raw')
	# smooth the data
#	a = smooth(a,winlen=5,lencorrect=True)
	# plot the smoothed data
#	pyplot.plot(a,'y', label='pre-smooth')
	# interpolate 'blinks' (artificial, due to smoothing of fake data and missing)
#	a = interpolate_blink(a, mode='auto', velthresh=5, maxdur=500, margin=10)
	# plot interpolated data
#	pyplot.plot(a,'b', label='blinks_interpolated')
	# interpolate missing data
	a = interpolate_missing(a,mode='linear',invalid=INVAL)
	# plot interpolated data
	pyplot.plot(a,'g', label='missing_interpolated')
	# remove outliers
	a = remove_outliers(a, maxdev=5, invalid=INVAL, interpolate=True, mode='auto')
	# plot data without outliers
	pyplot.plot(a,'m', label='outliers_removed')
	# smooth the data
	a = smooth(a,winlen=5,window='hanning',lencorrect=True)
	# plot the smoothed data
	pyplot.plot(a,'r', label='smooth')
	# finish the plot
	pyplot.legend(loc='upper right')
	pyplot.show()
# # # # #