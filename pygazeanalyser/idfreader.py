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

# IDF Reader
#
# Reads ASCII files as produced by SensoMotoric System's IDF converter,
# and performs a very crude fixation and blink detection: every sample that
# is invalid (usually coded '0.0') is considered to be part of a blink, and
# every sample in which the gaze movement velocity is below a threshold is
# considered to be part of a fixation. For optimal event detection, it would be
# better to use a different algorithm, e.g.:
# Nystrom, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
# saccade, and glissade detection in eyetracking data. Behavior Research
# Methods, 42, 188-204. doi:10.3758/BRM.42.1.188
#
# (C) Edwin Dalmaijer, 2014-2015
# edwin.dalmaijer@psy.ox.ax.uk
#
# version 1 (10-Jan-2015)

__author__ = "Edwin Dalmaijer"


import copy
import os.path

import numpy

from detectors import blink_detection, fixation_detection, saccade_detection


def read_idf(filename, start, stop=None, missing=0.0, debug=False):
	
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
	filestarted = False
	
	# loop through all lines
	for i in range(len(raw)):
		
		# string to list
		line = raw[i].replace('\n','').replace('\r','').split('\t')
		
		# check if the line starts with '##' (denoting header)
		if '##' in line[0]:
			# skip processing
			continue
		elif '##' not in line[0] and not filestarted:
			# check the indexes for several key things we want to extract
			# (we need to do this, because ASCII outputs of the IDF reader
			# are different, based on whatever the user wanted to extract)
			timei = line.index("Time")
			typei = line.index("Type")
			msgi = -1
			xi = {'L':None, 'R':None}
			yi = {'L':None, 'R':None}
			sizei = {'L':None, 'R':None}
			if "L POR X [px]" in line:
				xi['L']  = line.index("L POR X [px]")
			if "R POR X [px]" in line:
				xi['R']  = line.index("R POR X [px]")
			if "L POR Y [px]" in line:
				yi['L']  = line.index("L POR Y [px]")
			if "R POR Y [px]" in line:
				yi['R']  = line.index("R POR Y [px]")
			if "L Dia X [px]" in line:
				sizei['L']  = line.index("L Dia X [px]")
			if "R Dia X [px]" in line:
				sizei['R']  = line.index("R Dia X [px]")
			# set filestarted to True, so we don't attempt to extract
			# this info on all consecutive lines
			filestarted = True

		# check if trial has already started
		if started:
			# only check for stop if there is one
			if stop != None:
				if (line[typei] == 'MSG' and stop in line[msgi]) or i == len(raw)-1:
					started = False
					trialend = True
			# check for new start otherwise
			else:
				if start in line or i == len(raw)-1:
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
			if line[typei] == "MSG":
				if start in line[msgi]:
					message("trialstart %d" % len(data))
					# set started to True
					started = True
					# find starting time
					starttime = int(line[timei])
		
		# # # # #
		# parse line
		
		if started:
			# message lines will usually start with a timestamp, followed
			# by 'MSG', the trial number and the actual message, e.g.:
			#	"7818328012	MSG	1	# Message: 3"
			if line[typei] == "MSG":
				t = int(line[timei]) # time
				m = line[msgi] # message
				events['msg'].append([t,m])
			
			# regular lines will contain tab separated values, beginning with
			# a timestamp, follwed by the values that were chosen to be
			# extracted by the IDF converter
			else:
				# see if current line contains relevant data
				try:
					# extract data on POR and pupil size
					for var in ['x', 'y', 'size']:
						exec("vi = %si" % var)
						exec("v = %s" % var)
						# nothing
						if vi['L'] == None and vi['R'] == None:
							val = 'not in IDF'
						# only left eye
						elif vi['L'] != None and vi['R'] == None:
							val = float(line[vi['L']])
						# only right eye
						elif vi['L'] == None and vi['R'] != None:
							val = float(line[vi['R']])
						# average the two eyes, but only if they both
						# contain valid data
						elif vi['L'] != None and vi['R'] != None:
							if float(line[vi['L']]) == 0:
								val = float(line[vi['R']])
							elif float(line[vi['R']]) == 0:
								val = float(line[vi['L']])
							else:
								val = (float(line[vi['L']]) + float(line[vi['R']])) / 2.0
						v.append(val)
					# extract time data
					time.append(int(line[timei])-starttime)
					trackertime.append(int(line[timei]))
				except:
					message("line '%s' could not be parsed" % line)
					continue # skip this line	
	
	# # # # #
	# return
	
	return data
