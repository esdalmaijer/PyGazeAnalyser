# -*- coding: utf-8 -*-
#
# This file is part of PyGaze - the open-source toolbox for eye tracking
#
#    PyGazeAnalyser is a Python module for easily analysing eye-tracking data
#    Copyright (C) 2014  Edwin S. Dalmaijer
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>

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

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import nanmean

REPLACE_MISSINGS = 1
INTERPOLATE_MISSINGS = 2

def blink_detection(x, y, time, missing=0.0, minlen=10):

    """Detects blinks, defined as a period of missing data that lasts for at
    least a minimal amount of samples

    arguments

    x        -    np array of x positions
    y        -    np array of y positions
    time     -    np array of EyeTribe timestamps

    keyword arguments

    missing    -    value to be used for missing data (default = 0.0)
    minlen    -    integer indicating the minimal amount of consecutive
                missing samples

    returns
    Sblk, Eblk
                Sblk    -    list of lists, each containing [starttime]
                Eblk    -    list of lists, each containing [starttime,
                                                    endtime, duration]
    """

    # empty list to contain data
    Sblk = []
    Eblk = []

    # check where the missing samples are
    mx = np.array(x==missing, dtype=int)
    my = np.array(y==missing, dtype=int)
    miss = np.array((mx+my) == 2, dtype=int)

    # check where the starts and ends are (+1 to counteract shift to left)
    diff = np.diff(miss)
    starts = np.where(diff==1)[0] + 1
    ends = np.where(diff==-1)[0] + 1

    # compile blink starts and ends
    for i in range(len(starts)):
        # get starting index
        s = starts[i]
        # get ending index
        if i < len(ends):
            e = ends[i]
        elif len(ends) > 0:
            e = ends[-1]
        else:
            e = -1
        # append only if the duration in samples is equal to or greater than
        # the minimal duration
        if e-s >= minlen:
            # add starting time
            Sblk.append([time[s]])
            # add ending time
            Eblk.append([time[s],time[e],time[e]-time[s]])

    return Sblk, Eblk


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50,
                                handle_missings=None):

    """Detects fixations, defined as consecutive samples with an inter-sample
    distance of less than a set amount of pixels (disregarding missing data)

    arguments

    x        -    np array of x positions
    y        -    np array of y positions
    time        -    np array of EyeTribe timestamps

    keyword arguments

    missing    -    value to be used for missing data (default = 0.0)
    maxdist    -    maximal inter sample distance in pixels (default = 25)
    mindur     -    minimal duration of a fixation in milliseconds; detected
                fixation cadidates will be disregarded if they are below
                this duration (default = 100)

    handle_missings -  method to handle missings
                    if `REPLACE_MISSINGS`, missings will be replaced with the
                        previous sample
                    if `INTERPOLATE_MISSINGS`, missings will be interpolated

    returns
    Sfix, Efix
          Sfix    -    list of lists, each containing [starttime]
          Efix    -    list of lists, each containing [starttime,
                                            endtime, duration, endx, endy]
    """

    if handle_missings > 0:
        interpolate = handle_missings==INTERPOLATE_MISSINGS
        x = replace_missings(data = x, missing = missing,
                interpolate = interpolate)
        y = replace_missings(data = y, missing = missing,
                    interpolate = interpolate)

    # empty list to contain data
    Sfix = []
    Efix = []

    # loop through all coordinates
    si = 0
    fixstart = False
    for i in range(1,len(x)):
        # calculate Euclidean distance from the current fixation coordinate
        # to the next coordinate
        dist = ((x[si]-x[i])**2 + (y[si]-y[i])**2)**0.5
        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            Sfix.append([time[i]])
        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i-1]-Sfix[-1][0] >= mindur:
                Efix.append([Sfix[-1][0], time[i-1], time[i-1]-Sfix[-1][0],
                                            x[si], y[si]])
            # delete the last fixation start if it was too short
            else:
                Sfix.pop(-1)
            si = 0 + i
        elif not fixstart:
            si += 1

    return Sfix, Efix


def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40,
                    maxacc=340, handle_missings=None):

    """Detects saccades, defined as consecutive samples with an inter-sample
    velocity of over a velocity threshold or an acceleration threshold

    arguments

    x        -    np array of x positions
    y        -    np array of y positions
    time        -    np array of tracker timestamps in milliseconds

    keyword arguments

    missing    -    value to be used for missing data (default = 0.0)
    minlen    -    minimal length of saccades in milliseconds; all detected
                saccades with len(sac) < minlen will be ignored
                (default = 5)
    maxvel    -    velocity threshold in pixels/second (default = 40)
    maxacc    -    acceleration threshold in pixels / second**2
                (default = 340)
    handle_missings -  method to handle missings
                    if `REPLACE_MISSINGS`, missings will be replaced with the
                        previous sample
                    if `INTERPOLATE_MISSINGS`, missings will be interpolated


    returns
    Ssac, Esac
            Ssac    -    list of lists, each containing [starttime]
            Esac    -    list of lists, each containing [starttime, endtime,
                                        duration, startx, starty, endx, endy]
    """

    if handle_missings > 0:
        interpolate = handle_missings==INTERPOLATE_MISSINGS
        x = replace_missings(data = x, missing = missing,
                interpolate = interpolate)
        y = replace_missings(data = y, missing = missing,
                    interpolate = interpolate)


    # CONTAINERS
    Ssac = []
    Esac = []

    # INTER-SAMPLE MEASURES
    # the distance between samples is the square root of the sum
    # of the squared horizontal and vertical interdistances
    intdist = np.hypot(np.diff(x), np.diff(y))
        # get inter-sample times in seconds
    inttime = np.diff(time) / 1000.0

    # VELOCITY AND ACCELERATION
    # the velocity between samples is the inter-sample distance
    # divided by the inter-sample time
    vel = intdist / inttime
    # the acceleration is the sample-to-sample difference in
    # eye movement velocity
    acc = np.diff(vel)

    # SACCADE START AND END
    t0i = 0
    stop = False
    while not stop:
        # saccade start (t1) is when the velocity or acceleration
        # surpass threshold, saccade end (t2) is when both return
        # under threshold

        # detect saccade starts
        sacstarts = np.where((vel[1+t0i:] > maxvel) |
                                (acc[t0i:] > maxacc))[0]
        if len(sacstarts) > 0:
            # timestamp for starting position
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time)-1:
                t1i = len(time)-2
            t1 = time[t1i]

            # add to saccade starts
            Ssac.append([t1])

            # detect saccade endings
            sacends = np.where((vel[1+t1i:] < maxvel) &
                                  (acc[t1i:] < maxacc))[0]
            if len(sacends) > 0:
                # timestamp for ending position
                t2i = sacends[0] + 1 + t1i + 2
                if t2i >= len(time):
                    t2i = len(time)-1
                t2 = time[t2i]
                dur = t2 - t1

                # ignore saccades that did not last long enough
                if dur >= minlen:
                    # add to saccade ends
                    Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                else:
                    # remove last saccade start on too low duration
                    Ssac.pop(-1)

                # update t0i
                t0i = 0 + t2i
            else:
                stop = True
        else:
            stop = True

    return Ssac, Esac



def EK_microsaccade_detection(x,y,times, minimum_duration=12):
    """Engbert & Kliegl (2003) microsaccades detection algorithm
    http://www.sciencedirect.com/science/article/pii/S0042698903000841#

    (monocular detection only)
    """

    xy = np.transpose([x,y])
    dt = float(np.median(np.diff(times)))
    # moving average (5 samples) of velocity vectors
    vn = (np.roll(xy, -2, axis=0) + np.roll(xy, -1, axis=0) - \
         np.roll(xy, 1, axis=0)  - np.roll(xy, 2, axis=0)) / (6 * dt)
    vn[:2,:] = np.nan
    vn[-2:,:] = np.nan
    # criteria
    sigma = np.median(vn**2, axis=0) - np.median(vn, axis=0)**2
    eta = 6* sigma
    # find movements
    # Esac start, stop, duration, xstart, ystart, xend, yend
    Esac = []
    is_moving = False
    for c, m in enumerate(np.sum(vn > eta, axis=1) >0):
        if m:
            if not is_moving:
                Esac.append([times[c], np.nan, np.nan, xy[c,0], xy[c,1], np.nan, np.nan])
                is_moving = True
        else:
            if is_moving:
                Esac[-1][1] = times[c]
                Esac[-1][5] = xy[c,0]
                Esac[-1][6] = xy[c,1]
                is_moving = False
    Esac = np.array(Esac)
    Esac[:,2] =  Esac[:,1] - Esac[:,0] # duration
    # min samples 12 msec
    idx = np.where(Esac[:,2] >= minimum_duration)[0]
    Esac = Esac[idx, :]
    return Esac

def interpolate_missings(data, missing):
    """Interpolate missing values
    If the first sample is a missing, it will be replaced by np.nan.

    returns: np.array(dtype=float)

    """

    data = np.array(data, dtype=float)
    data[data == missing] = np.nan

    idx = np.arange(data.shape[0])
    not_nan = np.where(np.isfinite(data))
    f = interp1d(idx[not_nan], data[not_nan],bounds_error=False)
    rtn = np.where(np.isfinite(data),data,f(idx))

    return rtn

def replace_missings(data, missing, interpolate=False):
    """Replaces missing values with previous sample.
    If the first sample is a missing, it will be replaced by np.nan.
    If interpolation is true, values will be replaced by interpolated data.

    returns: np.array(dtype=float)

    """

    if interpolate:
       return interpolate_missings(data, missing)

    data = np.array(data, dtype=float)

    if data[0]==missing:
        data[0] = np.nan

    while True:
        idx = np.where(data == missing)[0]
        if len(idx) == 0:
            break
        data[idx] = data[idx-1]

    return data
