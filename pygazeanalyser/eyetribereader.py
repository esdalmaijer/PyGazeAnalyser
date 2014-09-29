# -*- coding: utf-8 -*-
#
# This file is part of PyGaze - the open-source toolbox for eye tracking
#
#	PyGazeAnalyser is a Python module for easily analysing eye-tracking data
#	Copyright (C) 2014  Edwin S. Dalmaijer
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>

# EyeTribe Reader
#
# Reads files as produced by PyTribe (https://github.com/esdalmaijer/PyTribe),
# and performs a very crude fixation and blink detection: every sample that
# is invalid (usually coded '0.0') is considered to be part of a blink, and
# every sample in which the gaze movement velocity is below a threshold is
# considered to be part of a fixation. For optimal event detection, it would be
# better to use a different algorithm, e.g.:
# Nystrom, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
# saccade, and glissade detection in eyetracking data. Behavior Research
# Methods, 42, 188-204. doi:10.3758/BRM.42.1.188
#
# (C) Edwin Dalmaijer, 2014
# edwin.dalmaijer@psy.ox.ax.uk
#
# version 1 (01-Jul-2014)

__author__ = "Edwin Dalmaijer"


import copy
import os.path

import numpy

from detectors import blink_detection, fixation_detection, saccade_detection


def read_eyetribe(filename, start, stop=None, missing=0.0, debug=False):
	
	"""Returns a list with dicts for every trial. A trial dict contains the
	following keys:
		x		-	numpy array of x positions
		y		-	numpy array of y positions
		size		-	numpy array of pupil size
		time		-	numpy array of timestamps, t=0 at trialstart
		trackertime-	numpy array of timestamps, according to the tracker
		events	-	dict with the following keys:
						Sfix	-	list of lists, each containing [starttime]
						Ssac	-	EMPTY! list of lists, each containing [starttime]
						Sblk	-	list of lists, each containing [starttime]
						Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
						Esac	-	EMPTY! list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
						Eblk	-	list of lists, each containing [starttime, endtime, duration]
						msg	-	list of lists, each containing [time, message]
						NOTE: timing is in EyeTribe time!
	
	arguments

	filename		-	path to the file that has to be read
	start		-	trial start string
	
	keyword arguments

	stop		-	trial ending string (default = None)
	missing	-	value to be used for missing data (default = 0.0)
	debug	-	Boolean indicating if DEBUG mode should be on or off;
				if DEBUG mode is on, information on what the script
				currently is doing will be printed to the console
				(default = False)
	
	returns

	data		-	a list with a dict for every trial (see above)
	"""

	# # # # #
	# debug mode
	
	if debug:
		def message(msg):
			print(msg)
	else:
		def message(msg):
			pass
		
	
	# # # # #
	# file handling
	
	# check if the file exists
	if os.path.isfile(filename):
		# open file
		message("opening file '%s'" % filename)
		f = open(filename, 'r')
	# raise exception if the file does not exist
	else:
		raise Exception("Error in read_eyetribe: file '%s' does not exist" % filename)
	
	# read file contents
	message("reading file '%s'" % filename)
	raw = f.readlines()
	
	# close file
	message("closing file '%s'" % filename)
	f.close()

	
	# # # # #
	# parse lines
	
	# variables
	data = []
	x = []
	y = []
	size = []
	time = []
	trackertime = []
	events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
	starttime = 0
	started = False
	trialend = False
	
	# loop through all lines
	for i in range(len(raw)):
		
		# string to list
		line = raw[i].replace('\n','').replace('\r','').split('\t')
		
		# check if trial has already started
		if started:
			# only check for stop if there is one
			if stop != None:
				if (line[0] == 'MSG' and stop in line[3]) or i == len(raw)-1:
					started = False
					trialend = True
			# check for new start otherwise
			else:
				if start in line:
					started = True
					trialend = True

			# # # # #
			# trial ending
			
			if trialend:
				message("trialend %d; %d samples found" % (len(data),len(x)))
				# trial dict
				trial = {}
				trial['x'] = numpy.array(x)
				trial['y'] = numpy.array(y)
				trial['size'] = numpy.array(size)
				trial['time'] = numpy.array(time)
				trial['trackertime'] = numpy.array(trackertime)
				trial['events'] = copy.deepcopy(events)
				# events
				trial['events']['Sblk'], trial['events']['Eblk'] = blink_detection(trial['x'],trial['y'],trial['trackertime'],missing=missing)
				trial['events']['Sfix'], trial['events']['Efix'] = fixation_detection(trial['x'],trial['y'],trial['trackertime'],missing=missing)
				trial['events']['Ssac'], trial['events']['Esac'] = saccade_detection(trial['x'],trial['y'],trial['trackertime'],missing=missing)
				# add trial to data
				data.append(trial)
				# reset stuff
				x = []
				y = []
				size = []
				time = []
				trackertime = []
				events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
				trialend = False
				
		# check if the current line contains start message
		else:
			if line[0] == "MSG":
				if start in line[3]:
					message("trialstart %d" % len(data))
					# set started to True
					started = True
					# find starting time
					starttime = int(line[2])
		
		# # # # #
		# parse line
		
		if started:
			# message lines will start with MSG, followed by a tab, then a
			# timestamp, a tab, the time, a tab and the message, e.g.:
			#	"MSG\t2014-07-01 17:02:33.770\t853589802\tsomething of importance here"
			if line[0] == "MSG":
				t = int(line[2]) # time
				m = line[3] # message
				events['msg'].append([t,m])
			
			# regular lines will contain tab separated values, beginning with
			# a timestamp, follwed by the values that were asked to be stored
			# in the data file. Usually, this comes down to
			# timestamp, time, fix, state, rawx, rawy, avgx, avgy, psize, 
			# Lrawx, Lrawy, Lavgx, Lavgy, Lpsize, Lpupilx, Lpupily,
			# Rrawx, Rrawy, Ravgx, Ravgy, Rpsize, Rpupilx, Rpupily
			# e.g.:
			# '2014-07-01 17:02:33.770, 853589802, False, 7, 512.5897, 510.8104, 614.6975, 614.3327, 16.8657,
			# 523.3592, 475.2756, 511.1529, 492.7412, 16.9398, 0.4037, 0.5209,
			# 501.8202, 546.3453, 609.3405, 623.2287, 16.7916, 0.5539, 0.5209'
			else:
				# see if current line contains relevant data
				try:
					# extract data
					x.append(float(line[6]))
					y.append(float(line[7]))
					size.append(float(line[8]))
					time.append(int(line[1])-starttime)
					trackertime.append(int(line[1]))
				except:
					message("line '%s' could not be parsed" % line)
					continue # skip this line	
	
	# # # # #
	# return
	
	return data
