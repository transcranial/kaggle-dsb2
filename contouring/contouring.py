#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import re
import dicom
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from natsort import natsorted
import errno


class EventHandler:
    def __init__(self, fpath, figure, axis):
        self.fpath = fpath
        self.figure = figure
        self.axis = axis
        self.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.figure.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.figure.canvas.mpl_connect('close_event', self.onclose)
        self.points = []
        self.lines = []
        self.coords = []

    def onpress(self, event):
        if event.inaxes != self.axis:
            return
        xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
        self.coords.append([yi, xi])
        self.points.extend(self.axis.plot([xi], [yi], 'ro'))
        if len(self.coords) > 1:
            self.lines.extend(self.axis.plot([xi, self.coords[-2][1]], [yi, self.coords[-2][0]], 'r-'))
        self.figure.canvas.draw()

    def onkeypress(self, event):
        if event.key == 'b':
            try:
                self.axis.lines.remove(self.points[-1])
                self.points.pop(-1)
                if len(self.coords) > 1:
                    self.axis.lines.remove(self.lines[-1])
                    self.lines.pop(-1)
                self.figure.canvas.draw()
                self.coords.pop(-1)
            except IndexError:
                pass

    def onclose(self, event):
        foutname = self.fpath.replace('.dcm', '.csv').replace('study/', '').replace('train', 'contours')
        try:
            os.makedirs(os.path.dirname(foutname))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(foutname)):
                pass
            else: raise
        with open(foutname, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.coords)
            print('[{}] written to file.'.format(foutname))

if __name__ == '__main__':
    try:
        dataset, phase = sys.argv[1], sys.argv[2]
    except IndexError:
        print('Not enough arguments. Specifiy the dataset number and the phase (ED or ES).')
        exit(0)

    if int(dataset) < 1 or int(dataset) > 500:
        raise Exception('Dataset specified, which is the first argument, is not valid.')

    if phase != 'ED' and phase != 'ES':
        raise Exception('The second argument, phase, must be specified as ES or ED.')

    with open('filepaths_train.json', 'r') as f:
        filepaths_train = json.load(f)

    with open('training_ed_es_slice_numbers.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        ed_es_slices = {}
        for study, ed, es in reader:
            ed_es_slices[study] = {'ED': int(ed), 'ES': int(es)}

    try:
        instance_num = ed_es_slices[dataset][phase]
    except KeyError:
        print('Dataset not present in the ED/ES csv file.')
        exit(0)

    # get all slices at instance number
    slice_fpaths = []
    for series in natsorted(filepaths_train[dataset].keys()):
        if not re.match(r'^sax_', series):
            continue
        if len(filepaths_train[dataset][series]) <= 30:
            for fname, fpath in filepaths_train[dataset][series]:
                if re.compile('^IM-\d+-{}.dcm$'.format(str(instance_num).zfill(4))).match(fname):
                    slice_fpaths.append(fpath)
                    break
        else:
            for fname, fpath in filepaths_train[dataset][series]:
                if re.compile('^IM-\d+-{}-\d+.dcm$'.format(str(instance_num).zfill(4))).match(fname):
                    slice_fpaths.append(fpath)


    for n, slice_fpath in enumerate(slice_fpaths):
        foutname = slice_fpath.replace('.dcm', '.csv').replace('study/', '').replace('train', 'contours')
        if os.path.isfile(foutname):
            print('[{}] already exists, skipping.'.format(foutname))
            continue

        im = plt.imshow(dicom.read_file(slice_fpath).pixel_array, interpolation='nearest', cmap=plt.cm.bone)
        fig = plt.gcf()
        ax = plt.gca()
        plt.title('dataset: {}, phase: {}\nslice filepath: {}\nwrite to contour filepath: {}'.format(dataset, phase, slice_fpath, slice_fpath.replace('.dcm', '.csv').replace('study/', '').replace('train', 'contours')), fontsize=10)
        ax.set_autoscale_on(False)

        handler = EventHandler(slice_fpath, fig, ax)
        plt.show()
