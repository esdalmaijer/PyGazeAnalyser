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

# OpenGaze Reader
#
# Reads files as produced by PyOpenGaze (https://github.com/esdalmaijer/PyOpenGaze),
# and performs a very crude fixation and blink detection: every sample that
# is invalid (usually coded '0.0') is considered to be part of a blink, and
# every sample in which the gaze movement velocity is below a threshold is
# considered to be part of a fixation. For optimal event detection, it would be
# better to use a different algorithm, e.g.:
# Nystrom, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
# saccade, and glissade detection in eyetracking data. Behavior Research
# Methods, 42, 188-204. doi:10.3758/BRM.42.1.188
#
# (C) Edwin Dalmaijer, 2017
# edwin.dalmaijer@psy.ox.ax.uk
#
# version 1 (20-Jun-2017)

__author__ = "Edwin Dalmaijer"


import copy
import os.path

import numpy

from detectors import blink_detection, fixation_detection, saccade_detection


def read_opengaze(filename, start, stop=None, missing=0.0, debug=False):
	
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
	
	# Parse the header.
	header = raw.pop(0)
	header = header.replace('\n','').replace('\r','').split('\t')
	
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
	laststart = None
	
	# loop through all lines
	for i in range(len(raw)):
		
		# string to list
		line = raw[i].replace('\n','').replace('\r','').split('\t')
		
		# check if trial has already started
		if started:
			# only check for stop if there is one
			if stop != None:
				if (line[header.index("USER")] != '0' and \
					stop in line[header.index("USER")]) \
					or i == len(raw)-1:
					started = False
					trialend = True
			# check for new start otherwise
			else:
				if start in line[header.index("USER")] \
					or i == len(raw)-1:
					# Only start if the current start is more than 1
					# sample away from the previous start.
					if (laststart == None) or (i != laststart + 1):
						started = True
						trialend = True
					laststart = i

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
			if line[header.index("USER")] != '0':
				if start in line[header.index("USER")]:
					# Only start if the current start is more than 1
					# sample away from the previous start.
					if (laststart == None) or (i != laststart + 1):
						message("trialstart %d" % len(data))
						# set started to True
						started = True
						# find starting time
						starttime = int(1000 * float(line[header.index("TIME")]))
					laststart = i
		
		# # # # #
		# parse line
		
		if started:
			# Messages are encoded in the user variable, which otherwise
			# is '0'. NOTE: Sometimes messages repeat in consecutive
			# samples, by accident.
			if line[header.index("USER")] != '0':
				t = int(1000 * float(line[header.index("TIME")])) # time
				m = line[header.index("USER")] # message
				events['msg'].append([t,m])
			
			# All lines (when obtained through PyOpenGaze or PyGaze)
			# should contain the following data:
			# 	CNT, TIME, TIME_TICK, 
			# 	FPOGX, FPOGY, FPOGS, FPOGD, FPOGID, FPOGV,
			# 	LPOGX, LPOGY, LPOGV, RPOGX, RPOGY, RPOGV,
			# 	BPOGX, BPOGY, BPOGV,
			# 	LPCX, LPCY, LPD, LPS, LPV, RPCX, RPCY, RPD, RPS, RPV
			# 	LEYEX, LEYEY, LEYEZ, LPUPILD, LPUPILV,
			# 	REYEX, REYEY, REYEZ, RPUPILD, RPUPILV,
			# 	CX, CY, CS, USER
			try:
				# Compute the size of the pupil.
				left = line[header.index("LPV")] == '1'
				right = line[header.index("RPV")] == '1'
				if left and right:
					s = (float(line[header.index("LPD")]) + \
						float(line[header.index("RPD")])) / 2.0
				elif left and not right:
					s = float(line[header.index("LPD")])
				elif not left and right:
					s = float(line[header.index("RPD")])
				else:
					s = 0.0
				# extract data
				x.append(float(line[header.index("BPOGX")]))
				y.append(float(line[header.index("BPOGY")]))
				size.append(s)
				time.append(int(1000 * float(line[header.index("TIME")]))-starttime)
				trackertime.append(int(1000 * float(line[header.index("TIME")])))
			except:
				message("line '%s' could not be parsed" % line)
				continue # skip this line	
	
	# # # # #
	# return
	
	return data
