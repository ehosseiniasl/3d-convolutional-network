#!/usr/bin/python
"""
3D-CAE with max-pooling

Stacked 3D-CAE for Alzheimer

11-11-15 Ehsan Hosseini-Asl

"""
__author__ = 'ehsanh'

import numpy as np
import argparse
import os
import pickle
import random
import sys
import time
import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, \
    roc_curve, auc, roc_auc_score
from convnet_3d import CAE3d, stacked_CAE3d
FLOAT_PRECISION = np.float32



def load_batch(batch_idx, num_batches, image_shape, data_dir, data_list):
    batch_size, d, c, h, w = image_shape
    if batch_idx<num_batches-1:
        batch_list = data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_data = np.empty(image_shape, dtype=FLOAT_PRECISION)
    else:
        batch_list = data_list[batch_idx*batch_size:]
        batch_data = np.empty((len(batch_list),d,c,h,w), dtype=FLOAT_PRECISION)
    labels = []
    for index, filename in enumerate(batch_list):
        data = sio.loadmat(data_dir+filename)
        data = data['original'].reshape(1, d, c, h, w)
        data_scaled = (data-data.min())/data.max()
        if 'AD' in filename:
            labels.append(0)
        elif 'MCI' in filename:
            labels.append(1)
        elif 'Normal' in filename:
            labels.append(2)
        else:
            raise NameError('filename is not ADNI subject')
        batch_data[index] = data_scaled
    return batch_data, labels, batch_list


def load_batch_fold(batch_idx, num_batches, image_shape, fold, data_dir, data_list):
    batch_size, d, c, h, w = image_shape
    if batch_idx<num_batches-1:
        batch_list = data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_data = np.empty(image_shape, dtype=FLOAT_PRECISION)
    else:
        batch_list = data_list[batch_idx*batch_size:]
        batch_data = np.empty((len(batch_list),d,c,h,w), dtype=FLOAT_PRECISION)
    labels = []
    for index, filename in enumerate(batch_list):
        data = sio.loadmat(data_dir+filename)
        data = data['original'].reshape(1, d, c, h, w)
        data_scaled = (data-data.min())/data.max()
        if 'AD' in filename:
            labels.append(0)
        elif 'MCI' in filename:
            labels.append(1)
        elif 'Normal' in filename:
            labels.append(2)
        else:
            raise NameError('filename is not ADNI subject')
        batch_data[index] = data_scaled
    return batch_data, labels, batch_list


def load_batch_AD_Normal(batch_idx, num_batches, image_shape, data_dir, data_list):
    batch_size, d, c, h, w = image_shape
    if batch_idx<num_batches-1:
        batch_list = data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_data = np.empty(image_shape, dtype=FLOAT_PRECISION)
    else:
        batch_list = data_list[batch_idx*batch_size:]
        batch_data = np.empty((len(batch_list),d,c,h,w), dtype=FLOAT_PRECISION)
    labels = []
    for index, filename in enumerate(batch_list):
        data = sio.loadmat(data_dir+filename)
        data = data['original'].reshape(1, d, c, h, w)
        data_scaled = (data-data.min())/data.max()
        if 'AD' in filename:
            labels.append(0)
        elif 'Normal' in filename:
            labels.append(1)
        else:
            raise NameError('filename is not ADNI subject')
        batch_data[index] = data_scaled
    return batch_data, labels, batch_list


def load_batch_MCI_Normal(batch_idx, num_batches, image_shape, data_dir, data_list):
    batch_size, d, c, h, w = image_shape
    if batch_idx<num_batches-1:
        batch_list = data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_data = np.empty(image_shape, dtype=FLOAT_PRECISION)
    else:
        batch_list = data_list[batch_idx*batch_size:]
        batch_data = np.empty((len(batch_list),d,c,h,w), dtype=FLOAT_PRECISION)
    labels = []
    for index, filename in enumerate(batch_list):
        data = sio.loadmat(data_dir+filename)
        data = data['original'].reshape(1, d, c, h, w)
        data_scaled = (data-data.min())/data.max()
        if 'MCI' in filename:
            labels.append(0)
        elif 'Normal' in filename:
            labels.append(1)
        else:
            raise NameError('filename is not ADNI subject')
        batch_data[index] = data_scaled
    return batch_data, labels, batch_list


def load_batch_AD_MCI(batch_idx, num_batches, image_shape, data_dir, data_list):
    batch_size, d, c, h, w = image_shape
    if batch_idx<num_batches-1:
        batch_list = data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_data = np.empty(image_shape, dtype=FLOAT_PRECISION)
    else:
        # ipdb.set_trace()
        batch_list = data_list[batch_idx*batch_size:]
        batch_data = np.empty((len(batch_list),d,c,h,w), dtype=FLOAT_PRECISION)
    labels = []
    for index, filename in enumerate(batch_list):
        data = sio.loadmat(data_dir+filename)
        data = data['original'].reshape(1, d, c, h, w)
        data_scaled = (data-data.min())/data.max()
        if 'AD' in filename:
            labels.append(0)
        elif 'MCI' in filename:
            labels.append(1)
        else:
            raise NameError('filename is not ADNI subject')
        batch_data[index] = data_scaled
    return batch_data, labels, batch_list


def load_batch_AM_N(batch_idx, num_batches, image_shape, data_dir, data_list):
    batch_size, d, c, h, w = image_shape
    if batch_idx<num_batches-1:
        batch_list = data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_data = np.empty(image_shape, dtype=FLOAT_PRECISION)
    else:
        batch_list = data_list[batch_idx*batch_size:]
        batch_data = np.empty((len(batch_list),d,c,h,w), dtype=FLOAT_PRECISION)
    labels = []
    for index, filename in enumerate(batch_list):
        data = sio.loadmat(data_dir+filename)
        data = data['original'].reshape(1, d, c, h, w)
        data_scaled = (data-data.min())/data.max()
        if 'AD' in filename or 'MCI' in filename:
            labels.append(0)
        elif 'Normal' in filename:
            labels.append(1)
        else:
            raise NameError('filename is not ADNI subject')
        batch_data[index] = data_scaled
    return batch_data, labels, batch_list


def do_pretraining_cae(data_dir, models, cae_layer, max_epoch=1):
    data_list = os.listdir(data_dir)
    random.shuffle(data_list)
    batch_size, d, c, h, w = models[cae_layer-1].image_shape
    progress_report = 1
    save_interval = 1800
    num_subjects = len(data_list)
    num_batches = num_subjects/batch_size
    last_save = time.time()
    epoch = 0
    print 'training CAE_'+str(cae_layer)
    while True:
        try:
            loss = 0
            start_time = time.time()
            for batch in xrange(num_batches):
                batch_data, labels = load_batch(batch, models[0].image_shape, data_dir, data_list=data_list)
                start = time.time()
                if cae_layer == 1:
                    cost = models[0].train(batch_data)
                elif cae_layer == 2:
                    hidden_batch = models[0].get_activation(batch_data)
                    cost = models[1].train(hidden_batch)
                elif cae_layer == 3:
                    hidden_batch1 = models[0].get_activation(batch_data)
                    hidden_batch2 = models[1].get_activation(hidden_batch1)
                    cost = models[2].train(hidden_batch2)
                loss += cost
                train_time = time.time()-start
                print 'batch:%02d\tcost:%.2f\ttime:%.2f' % (batch, cost, train_time/60.)
                sys.stdout.flush()
            epoch += 1
            if epoch % progress_report == 0:
                print '%02d\t%g\t%f' % (epoch, loss, time.time()-start_time)
                sys.stdout.flush()
            if time.time() - last_save >= save_interval:
                filename = 'cae'+str(cae_layer)\
                           +'_[act=%s,fn=%d,fs=%d].pkl'%\
                            (models[cae_layer-1].activation, models[cae_layer-1].flt_channels, models[cae_layer-1].flt_width)
                models[cae_layer-1].save(filename)
                print 'model saved to', filename
                sys.stdout.flush()
                last_save = time.time()
            if epoch >= max_epoch-1:
                filename = 'cae'+str(cae_layer)\
                           +'_[act=%s,fn=%d,fs=%d].pkl'%\
                            (models[cae_layer-1].activation, models[cae_layer-1].flt_channels, models[cae_layer-1].flt_width)
                models[cae_layer-1].save(filename)
                print 'max epoch reached. model saved to', filename
                sys.stdout.flush()
                return filename

        except KeyboardInterrupt:
            # filename = 'cae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
            filename = 'cae'+str(cae_layer)\
                       +'_[act=%s,fn=%d,fs=%d].pkl'%\
                        (models[cae_layer-1].activation, models[cae_layer-1].flt_channels, models[cae_layer-1].flt_width)
            models[cae_layer-1].save(filename)
            print 'model saved to', filename
            sys.stdout.flush()
            return filename


def finetune_scae(data_dir, model, binary_classification=(False, False, False, False), max_epoch=1):
    data_list = os.listdir(data_dir)
    random.shuffle(data_list)
    batch_size, d, c, h, w = model.image_shape
    progress_report = 1
    save_interval = 1800
    num_subjects = len(data_list)
    num_batches = num_subjects/batch_size
    if num_subjects%batch_size !=0:
        num_batches  +=1
    last_save = time.time()
    epoch = 0
    AD_Normal, AD_MCI, MCI_Normal, AM_N = binary_classification

    if True not in binary_classification:
        filename = 'scae.pkl'
        print 'training scae for AD_MCI_Normal'
    elif not MCI_Normal and not AM_N:
        filename = 'scae_%s.pkl'%('AD_Normal' if AD_Normal else 'AD_MCI')
        print 'training scae for %s'%('AD_Normal' if AD_Normal else 'AD_MCI')
    else:
        filename = 'scae_%s.pkl'%('MCI_Normal' if MCI_Normal else 'AM_N')
        print 'training scae for %s'%('MCI_Normal' if MCI_Normal else 'AM_N')

    while True:
        try:
            loss_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            error_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            start_time = time.time()
            for batch in xrange(num_batches):
                if True not in binary_classification:
                    batch_data, batch_labels, batch_names = load_batch(batch, num_batches, model.image_shape, data_dir,
                                                          data_list=data_list)
                elif AD_Normal:
                    batch_data, batch_labels, batch_names = load_batch_AD_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                     data_list=data_list)
                elif AD_MCI:
                    batch_data, batch_labels, batch_names = load_batch_AD_MCI(batch, num_batches, model.image_shape, data_dir,
                                                                 data_list=data_list)
                elif MCI_Normal:
                    batch_data, batch_labels, batch_names = load_batch_MCI_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                 data_list=data_list)
                elif AM_N:
                    batch_data, batch_labels, batch_names = load_batch_AM_N(batch, num_batches, model.image_shape, data_dir,
                                                                 data_list=data_list)
                start = time.time()

                batch_error, cost, pred, prob = model.train(batch_data, batch_labels)
                loss_hist[batch] = cost
                train_time = time.time()-start
                print
                error_hist[batch] = batch_error
                print 'batch:%02d\terror:%.2f\tcost:%.2f\ttime:%.2f' % (batch, batch_error, cost, train_time/60.)
                print 'subjects:\t',
                for name in batch_names:
                    print name[:-4],
                print
                print 'labels:\t',
                for l in batch_labels:
                    print l,
                print
                print 'pred:\t',
                for p in pred:
                    print p,
                print
                print 'prob:',
                for p in prob:
                    print '%.2f'%p,
                print
                sys.stdout.flush()
            epoch += 1
            if epoch % progress_report == 0:
                print 'epoch:%02d\terror:%g\tloss:%g\ttime:%f' % (epoch, error_hist.mean(), loss_hist.mean(),
                                                                  (time.time()-start_time)/60.)
                sys.stdout.flush()
            if time.time() - last_save >= save_interval:
                model.save(filename)
                print 'scae model saved to', filename
                sys.stdout.flush()
                last_save = time.time()
            if epoch >= max_epoch-1:
                model.save(filename)
                print 'max epoch reached. scae model saved to', filename
                sys.stdout.flush()
                return filename
        except KeyboardInterrupt:
            model.save(filename)
            print 'scae model saved to', filename
            sys.stdout.flush()
            return filename


def finetune_scae_crossvalidate(data_dir, model,
                                binary_classification=(False, False, False, False),
                                max_epoch=1):
    data_list = os.listdir(data_dir)
    random.shuffle(data_list)
    batch_size, d, c, h, w = model.image_shape
    progress_report = 1
    save_interval = 1800
    num_subjects = int(4./5*len(data_list))
    num_batches = num_subjects/batch_size
    if num_subjects%batch_size !=0:
        num_batches  +=1
    last_save = time.time()
    AD_Normal, AD_MCI, MCI_Normal, AM_N = binary_classification

    for fold in xrange(2,5):
        epoch = 0
        model.layers[3].initialize_layer()
        model.layers[4].initialize_layer()
        model.layers[5].initialize_layer()

        data_list_fold = [data for data in data_list if int(data[-6:-4])%5!=fold]
        if True not in binary_classification:
            filename = 'scae_fold%d.pkl'%(fold)
            print 'training scae for AD_MCI_Normal'
        elif not MCI_Normal and not AM_N:
            filename = 'scae_%s_%d.pkl'%('AD_Normal' if AD_Normal else 'AD_MCI', fold)
            print 'training scae for %s, fold %d'%('AD_Normal' if AD_Normal else 'AD_MCI', fold)
        else:
            filename = 'scae_%s_fold%d.pkl'%('MCI_Normal' if MCI_Normal else 'AM_N', fold)
            print 'training scae for %s, fold %d'%('MCI_Normal' if MCI_Normal else 'AM_N', fold)

        error = 1
        while error>0.04:
            try:
                loss_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
                error_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
                start_time = time.time()
                for batch in xrange(num_batches):
                    if True not in binary_classification:
                        batch_data, batch_labels, batch_names = load_batch(batch, num_batches, model.image_shape, data_dir,
                                                                           data_list=data_list_fold)
                    elif AD_Normal:
                        batch_data, batch_labels, batch_names = load_batch_AD_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                                     data_list=data_list_fold)
                    elif AD_MCI:
                        batch_data, batch_labels, batch_names = load_batch_AD_MCI(batch, num_batches, model.image_shape, data_dir,
                                                                                  data_list=data_list_fold)
                    elif MCI_Normal:
                        batch_data, batch_labels, batch_names = load_batch_MCI_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                                      data_list=data_list_fold)
                    elif AM_N:
                        batch_data, batch_labels, batch_names = load_batch_AM_N(batch, num_batches, model.image_shape, data_dir,
                                                                                data_list=data_list_fold)
                    start = time.time()

                    batch_error, cost, pred, prob = model.train(batch_data, batch_labels)
                    loss_hist[batch] = cost
                    train_time = time.time()-start
                    print
                    error_hist[batch] = batch_error
                    print 'batch:%02d\terror:%.2f\tcost:%.2f\ttime:%.2f' % (batch, batch_error, cost, train_time/60.)
                    print 'subjects:\t',
                    for name in batch_names:
                        print name[:-4],
                    print
                    print 'labels:\t',
                    for l in batch_labels:
                        print l,
                    print
                    print 'pred:\t',
                    for p in pred:
                        print p,
                    print
                    print 'prob:',
                    for p in prob:
                        print '%.2f'%p,
                    print
                    sys.stdout.flush()
                epoch += 1
                error = error_hist.mean()
                if epoch % progress_report == 0:
                    print 'epoch:%02d\terror:%.2f\tloss:%.2f\ttime:%02d min' % (epoch, error_hist.mean(),
                                                                                loss_hist.mean(),
                                                                                (time.time()-start_time)/60.)
                    sys.stdout.flush()
                if time.time() - last_save >= save_interval:
                    model.save(filename)
                    print 'scae model fold %d saved to %s'% (fold, filename)
                    sys.stdout.flush()
                    last_save = time.time()
            except KeyboardInterrupt:
                model.save(filename)
                print 'scae model fold %d saved to %s'% (fold, filename)
                sys.stdout.flush()
                continue
        model.save(filename)
        print 'error threshold reached. scae model fold %d saved to %s' % (fold, filename)
        sys.stdout.flush()
        continue


def get_hidden_data(dir, image_shape, models, layer):
    ''' print 'get hidden activation '''
    hidden_data = {}
    for type in ['AD', 'MCI', 'Normal']:
        print 'get %s hidden activation' % type
        data_dir = dir+type+'/'
        data_list = os.listdir(data_dir)
        data_list.sort()
        num_subjects = len(data_list)
        batch_size = image_shape[0]
        num_batches = num_subjects/batch_size

        sample_batch, _, _ = load_batch(0, num_batches, image_shape, data_dir, data_list=data_list)
        if layer == 1:
            sample_hidden = models[layer-1].get_activation(sample_batch)
        elif layer == 2:
            hidden_batch = models[layer-2].get_activation(sample_batch)
            sample_hidden = models[layer-1].get_activation(hidden_batch)
        else:
            hidden1_batch = models[layer-3].get_activation(sample_batch)
            hidden2_batch = models[layer-2].get_activation(hidden1_batch)
            sample_hidden = models[layer-1].get_activation(hidden2_batch)
        sample_shape = sample_hidden.shape
        _, depth, channel, height, width = sample_shape
        hidden_shape = (len(data_list), depth, channel, height, width)

        hidden_data[type] = np.empty(hidden_shape, dtype=FLOAT_PRECISION)
        for batch in xrange(num_batches):
            print batch
            batch_data, _, _ = load_batch(batch, num_batches, image_shape, data_dir, data_list=data_list)
            if layer == 1:
                batch_hidden = models[layer-1].get_activation(batch_data)
            elif layer == 2:
                batch_hidden1 = models[layer-2].get_activation(batch_data)
                batch_hidden = models[layer-1].get_activation(batch_hidden1)
            else:
                batch_hidden1 = models[layer-3].get_activation(batch_data)
                batch_hidden2 = models[layer-2].get_activation(batch_hidden1)
                batch_hidden = models[layer-1].get_activation(batch_hidden2)
            hidden_data[type][batch*batch_size:(batch+1)*batch_size] = batch_hidden
        for i in xrange(10):
            filename = '%s_hidden_layer%d_%d.mat' % (type, layer, i)
            sio.savemat(filename, {'hidden_data':hidden_data[type][i*10:(i+1)*10]})
    return hidden_data


def get_hidden_finetuned(dir, model, layer):
    ''' print 'get hidden activation '''
    hidden_data = {}
    for type in ['AD', 'MCI', 'Normal']:
        print 'get %s hidden activation' % type
        sys.stdout.flush()
        data_dir = dir+type+'/'
        data_list = os.listdir(data_dir)
        data_list.sort()
        num_subjects = len(data_list)
        batch_size = model.image_shape[0]
        num_batches = num_subjects/batch_size

        sample_batch, _, _ = load_batch(0, num_batches, model.image_shape, data_dir, data_list=data_list)
        if layer == 1:
            sample_hidden = model.layers[layer-1].get_activation(sample_batch)
        elif layer == 2:
            hidden_batch = model.layers[layer-2].get_activation(sample_batch)
            sample_hidden = model.layers[layer-1].get_activation(hidden_batch)
        else:
            hidden1_batch = model.layers[layer-3].get_activation(sample_batch)
            hidden2_batch = model.layers[layer-2].get_activation(hidden1_batch)
            sample_hidden = model.layers[layer-1].get_activation(hidden2_batch)
        sample_shape = sample_hidden.shape
        _, depth, channel, height, width = sample_shape
        hidden_shape = (len(data_list), depth, channel, height, width)

        hidden_data[type] = np.empty(hidden_shape, dtype=FLOAT_PRECISION)
        for batch in xrange(num_batches):
            batch_data, _, _ = load_batch(batch, num_batches, model.image_shape, data_dir, data_list=data_list)
            start_time = time.time()
            if layer == 1:
                batch_hidden = model.layers[layer-1].get_activation(batch_data)
            elif layer == 2:
                batch_hidden1 = model.layers[layer-2].get_activation(batch_data)
                batch_hidden = model.layers[layer-1].get_activation(batch_hidden1)
            else:
                batch_hidden1 = model.layers[layer-3].get_activation(batch_data)
                batch_hidden2 = model.layers[layer-2].get_activation(batch_hidden1)
                batch_hidden = model.layers[layer-1].get_activation(batch_hidden2)
            forward_time = time.time()-start_time
            print 'batch:%d\ttime: %.2f min' % (batch, forward_time/60.)
            hidden_data[type][batch*batch_size:(batch+1)*batch_size] = batch_hidden
            sys.stdout.flush()
        for i in xrange(10):
            filename = 'hidden_layer%d/%s_hidden_layer%d_%d.mat' % (layer, type, layer, i)
            sio.savemat(filename, {'hidden_data':hidden_data[type][i*10:(i+1)*10]})
            sys.stdout.flush()
    return hidden_data


def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train scae on alzheimer')
    default_image_dir = 'ADNI_original/data/'
    parser.add_argument('-I', '--data_dir', default=default_image_dir,
                        help='location of image files; default=%s' % default_image_dir)
    parser.add_argument('-m', '--scae_model',
                        help='start with this scae model')
    parser.add_argument('-cae1', '--cae1_model',
                        help='Initialize cae1 model')
    parser.add_argument('-cae2', '--cae2_model',
                        help='Initialize cae2 model')
    parser.add_argument('-cae3', '--cae3_model',
                        help='Initialize cae3 model')
    parser.add_argument('-ac', '--activation_cae', type=str, default='relu',
                        help='cae activation function')
    parser.add_argument('-af', '--activation_final', type=str, default='relu',
                        help='final layer activation function')
    parser.add_argument('-fn', '--filter_channel', type=int, default=[8,8,8], nargs='+',
                        help='filter channel list')
    parser.add_argument('-fs', '--filter_size', type=int, default=3,
                        help='filter size')
    parser.add_argument('-p', '--pretrain_layer', type=int, default=0,
                        help='pretrain cae layer')
    parser.add_argument('-gh', '--get_hidden', type=int, default=0,
                        help='get hidden layer')
    parser.add_argument('-t', '--test', action='store_true',
                        help='do testing')
    parser.add_argument('-ft', '--finetune', action='store_true',
                        help='do fine tuning')
    parser.add_argument('-AN', '--AD_Normal', action='store_true',
                        help='AD-Normal classification')
    parser.add_argument('-AM', '--AD_MCI', action='store_true',
                        help='AD-MCI classification')
    parser.add_argument('-MN', '--MCI_Normal', action='store_true',
                        help='MCI-Normal classification')
    parser.add_argument('-AMN', '--AM_N', action='store_true',
                        help='AM-Normal classification')
    parser.add_argument('-lcn', '--load_conv', action='store_true',
                        help='load only conv layers')
    parser.add_argument('-batch', '--batchsize', type=int, default=1,
                        help='batch size')
    args = parser.parse_args()
    return args.data_dir, args.scae_model, args.cae1_model, args.cae2_model, args.cae3_model, args.activation_cae, \
           args.activation_final, \
           args.filter_channel, args.filter_size, args.pretrain_layer, args.get_hidden, args.test, \
           args.finetune, args.AD_Normal, args.AD_MCI, args.MCI_Normal, args.AM_N, args.load_conv, args.batchsize


def test_scae(data_dir, model, binary_classification=(False, False, False, False)):
    data_list = os.listdir(data_dir)
    batch_size, d, c, h, w = model.image_shape
    num_subjects = len(data_list)
    num_batches = num_subjects/batch_size
    if num_subjects%batch_size !=0:
        num_batches  +=1
    AD_Normal, AD_MCI, MCI_Normal, AM_N = binary_classification
    if True not in binary_classification:
        print 'testing scae for AD_MCI_Normal'
    elif not MCI_Normal and not AM_N:
        print 'testing scae for %s'%('AD_Normal' if AD_Normal else 'AD_MCI')
        filename = 'test_%s.pkl'%('AD_Normal' if AD_Normal else 'AD_MCI')
    else:
        filename = 'test_%s.pkl'%('MCI_Normal' if MCI_Normal else 'AM_N')
        print 'testing scae for %s'%('MCI_Normal' if MCI_Normal else 'AM_N')
    sys.stdout.flush()

    test_labels, test_names, test_pred, test_prob, test_label_prob= [], [], [], [], []
    num_labels = 2 if True in binary_classification else 3
    p_y_given_x = np.empty((num_subjects, num_labels), dtype=FLOAT_PRECISION)
    conv2_feat = np.empty((num_subjects, np.prod(model.conv2_output_shape[1:])), dtype=FLOAT_PRECISION)
    conv3_feat = np.empty((num_subjects, np.prod(model.conv3_output_shape[1:])), dtype=FLOAT_PRECISION)
    ip2_feat = np.empty((num_subjects, 500), dtype=FLOAT_PRECISION)
    ip1_feat = np.empty((num_subjects, 2000), dtype=FLOAT_PRECISION)
    image_gradient = np.empty((num_subjects, d, c, h, w), dtype=FLOAT_PRECISION)
    for batch in xrange(num_batches):
        if True not in binary_classification:
            batch_data, batch_labels, batch_names = load_batch(batch, num_batches, model.image_shape, data_dir,
                                                               data_list=data_list)
        elif AD_Normal:
            batch_data, batch_labels, batch_names = load_batch_AD_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                         data_list=data_list)
        elif AD_MCI:
            batch_data, batch_labels, batch_names = load_batch_AD_MCI(batch, num_batches, model.image_shape, data_dir,
                                                                      data_list=data_list)
        elif MCI_Normal:
            batch_data, batch_labels, batch_names = load_batch_MCI_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                          data_list=data_list)
        elif AM_N:
            batch_data, batch_labels, batch_names = load_batch_AM_N(batch, num_batches, model.image_shape, data_dir,
                                                                    data_list=data_list)
        batch_error, pred, prob, truth_prob, batch_p_y_given_x, batch_conv2_feat, \
        batch_conv3_feat, batch_ip2_feat, batch_ip1_feat, batch_gradient\
            = model.forward(batch_data, batch_labels)
        test_labels.extend(batch_labels)
        test_names.extend(batch_names)
        test_pred.extend(pred)
        test_prob.extend(prob)
        test_label_prob.extend(truth_prob)
        p_y_given_x[batch*batch_size:(batch+1)*batch_size, :] = batch_p_y_given_x
        conv2_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_conv2_feat
        conv3_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_conv3_feat
        ip2_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_ip2_feat
        ip1_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_ip1_feat
        image_gradient[batch*batch_size:(batch+1)*batch_size, :] = batch_gradient
        for i, subject in enumerate(batch_names):
            sio.savemat('{0}_gradient.mat'.format(subject[:-4]), {'gradient':batch_gradient[i]})

        print '\n\nbatch:%02d\terror:%.2f' % (batch, batch_error)
        print 'subjects:\t',
        for name in batch_names:
            print name[:-4],
        print
        print 'labels:\t',
        for l in batch_labels:
            print l,
        print
        print 'pred:\t',
        for p in pred:
            print p,
        print
        print 'prob:',
        for p in prob:
            print '%.2f'%p,
        print
        sys.stdout.flush()

    accuracy = accuracy_score(test_labels, test_pred)
    f_score = f1_score(np.asarray(test_labels), np.asarray(test_pred))
    confusion = confusion_matrix(np.asarray(test_labels), np.asarray(test_pred))

    print '\n\nAccuracy:%.4f\tF1_Score:%.4f' % (accuracy, f_score)
    print '\nconfusion:'
    print confusion

    if True not in binary_classification:
        class_names = ['AD', 'MCI', 'Normal']
        filename = 'test_AMN.pkl'
    elif AD_Normal:
        class_names = ['AD', 'Normal']
        filename = 'test_AN.pkl'
    elif AD_MCI:
        class_names = ['AD', 'MCI']
        filename = 'test_AM.pkl'
    elif MCI_Normal:
        class_names = ['MCI', 'Normal']
        filename = 'test_MN.pkl'
    elif AM_N:
        class_names = ['AD_MCI', 'Normal']
        filename = 'test_AM_N.pkl'

    results_report = classification_report(test_labels, test_pred, target_names=class_names)
    print '\nclassification report:'
    print results_report

    results = (test_names, test_labels, test_label_prob, test_pred, test_prob,
               p_y_given_x, results_report, class_names)

    f = open(filename, 'wb')
    pickle.dump(results, f, -1)
    f.close()
    f=open('image_gradient.pkl', 'wb')
    pickle.dump(image_gradient, f, -1)
    f.close()


def test_scae_crossvalidate(data_dir, model, binary_classification=(False, False, False, False)):
    data_list = os.listdir(data_dir)
    batch_size, d, c, h, w = model.image_shape
    num_subjects = int(1./5*len(data_list))
    num_batches = num_subjects/batch_size
    if num_subjects%batch_size != 0:
        num_batches += 1
    AD_Normal, AD_MCI, MCI_Normal, AM_N = binary_classification

    for fold in xrange(5):
        data_list_fold = [data for data in data_list if int(data[-6:-4])%5==fold]
        if True not in binary_classification:
            print 'testing scae for fold %d AD_MCI_Normal' % (fold)
        elif not MCI_Normal and not AM_N:
            print 'testing scae for %s for fold %d'%('AD_Normal' if AD_Normal else 'AD_MCI', fold)
            filename = 'scae_%s_fold%d.pkl'%('AD_Normal' if AD_Normal else 'AD_MCI', fold)
        else:
            filename = 'scae_%s_fold%d.pkl'%('MCI_Normal' if MCI_Normal else 'AM_N', fold)
            print 'testing scae for %s for fold %d'%('MCI_Normal' if MCI_Normal else 'AM_N', fold)
        model.load(filename)
        sys.stdout.flush()

        test_labels, test_names, test_pred, test_prob, test_label_prob= [], [], [], [], []
        num_labels = 2 if True in binary_classification else 3
        p_y_given_x = np.empty((num_subjects, num_labels), dtype=FLOAT_PRECISION)
        conv2_feat = np.empty((num_subjects, np.prod(model.conv2_output_shape[1:])), dtype=FLOAT_PRECISION)
        conv3_feat = np.empty((num_subjects, np.prod(model.conv3_output_shape[1:])), dtype=FLOAT_PRECISION)
        ip2_feat = np.empty((num_subjects, 500), dtype=FLOAT_PRECISION)
        ip1_feat = np.empty((num_subjects, 2000), dtype=FLOAT_PRECISION)
        for batch in xrange(num_batches):
            if True not in binary_classification:
                batch_data, batch_labels, batch_names = load_batch(batch, num_batches, model.image_shape, data_dir,
                                                                   data_list=data_list_fold)
            elif AD_Normal:
                batch_data, batch_labels, batch_names = load_batch_AD_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                             data_list=data_list_fold)
            elif AD_MCI:
                batch_data, batch_labels, batch_names = load_batch_AD_MCI(batch, num_batches, model.image_shape, data_dir,
                                                                          data_list=data_list_fold)
            elif MCI_Normal:
                batch_data, batch_labels, batch_names = load_batch_MCI_Normal(batch, num_batches, model.image_shape, data_dir,
                                                                              data_list=data_list_fold)
            elif AM_N:
                batch_data, batch_labels, batch_names = load_batch_AM_N(batch, num_batches, model.image_shape, data_dir,
                                                                        data_list=data_list_fold)
            batch_error, pred, prob, truth_prob, batch_p_y_given_x, batch_conv2_feat, \
            batch_conv3_feat, batch_ip2_feat, batch_ip1_feat, batch_gradient\
                = model.forward(batch_data, batch_labels)
            test_labels.extend(batch_labels)
            test_names.extend(batch_names)
            test_pred.extend(pred)
            test_prob.extend(prob)
            test_label_prob.extend(truth_prob)
            p_y_given_x[batch*batch_size:(batch+1)*batch_size, :] = batch_p_y_given_x
            conv2_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_conv2_feat
            conv3_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_conv3_feat
            ip2_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_ip2_feat
            ip1_feat[batch*batch_size:(batch+1)*batch_size, :] = batch_ip1_feat

            print '\n\nbatch:%02d\terror:%.2f' % (batch, batch_error)
            print 'subjects:\t',
            for name in batch_names:
                print name[:-4],
            print
            print 'labels:\t',
            for l in batch_labels:
                print l,
            print
            print 'pred:\t',
            for p in pred:
                print p,
            print
            print 'prob:',
            for p in prob:
                print '%.2f'%p,
            print
            sys.stdout.flush()

        accuracy = accuracy_score(test_labels, test_pred)
        f_score = f1_score(np.asarray(test_labels), np.asarray(test_pred))
        confusion = confusion_matrix(np.asarray(test_labels), np.asarray(test_pred))
        computed_auc = roc_auc_score(test_labels, test_pred)
        print '\n\nAccuracy:%.4f\tF1_Score:%.4f\tAUC:%.4f' % (accuracy, f_score, computed_auc)
        print '\nconfusion:'
        print confusion

        if True not in binary_classification:
            class_names = ['AD', 'MCI', 'Normal']
            filename = 'test_AMN_fold{0}.pkl'.format(fold)
        elif AD_Normal:
            class_names = ['AD', 'Normal']
            filename = 'test_AN_fold{0}.pkl'.format(fold)
        elif AD_MCI:
            class_names = ['AD', 'MCI']
            filename = 'test_AM_fold{0}.pkl'.format(fold)
        elif MCI_Normal:
            class_names = ['MCI', 'Normal']
            filename = 'test_MN_fold{0}.pkl'.format(fold)
        elif AM_N:
            class_names = ['AD_MCI', 'Normal']
            filename = 'test_AM_N_fold{0}.pkl'.format(fold)

        results_report = classification_report(test_labels, test_pred, target_names=class_names)
        print '\nclassification report:'
        print results_report

        results = (test_names, test_labels, test_label_prob, test_pred, test_prob,
                   p_y_given_x, results_report, class_names)

        f = open(filename, 'wb')
        pickle.dump(results, f, -1)
        f.close()


def main():
    data_dir, scae_model, cae1_model, cae2_model, cae3_model, activation_cae, activation_final, \
    flt_channels, flt_size, pretrain_layer, get_hidden, test, finetune, AD_Normal, AD_MCI, MCI_Normal, AM_N, load_conv, batchsize = \
        ProcessCommandLine()
    binary = (AD_Normal, AD_MCI, MCI_Normal, AM_N)
    print 'cae activation:', activation_cae
    print 'final layers activation:', activation_final
    print 'filter channels:', flt_channels
    print 'filter size:', flt_size
    sys.stdout.flush()
    data_list = os.listdir(data_dir)
    sample = sio.loadmat(data_dir+data_list[0])
    depth, height, width = sample['original'].shape
    in_channels   = 1
    in_time       = depth
    in_width      = width
    in_height     = height
    flt_depth     = flt_size
    flt_width     = flt_size
    flt_height    = flt_size

    image_shp = (batchsize, in_time, in_channels, in_height, in_width)
    filter_shp_1 = (flt_channels[0], flt_depth, in_channels, flt_height, flt_width)
    filter_shp_2 = (flt_channels[1], flt_depth, filter_shp_1[0], flt_height, flt_width)
    filter_shp_3 = (flt_channels[2], flt_depth, filter_shp_2[0], flt_height, flt_width)

    if not finetune and not test and get_hidden==0:
        cae1 = CAE3d(signal_shape=image_shp,
                     filter_shape=filter_shp_1,
                     poolsize=(2, 2, 2),
                     activation=activation_cae)
        print 'CAE1 built'
        if cae1_model:
            cae1.load(cae1_model)
        sys.stdout.flush()

        cae2 = CAE3d(signal_shape=cae1.hidden_pooled_image_shape,
                     filter_shape=filter_shp_2,
                     poolsize=(2, 2, 2),
                     activation=activation_cae)
        print 'CAE2 built'
        if cae2_model:
            cae2.load(cae2_model)
        sys.stdout.flush()

        cae3 = CAE3d(signal_shape=cae2.hidden_pooled_image_shape,
                     filter_shape=filter_shp_3,
                     poolsize=(2, 2, 2),
                     activation=activation_cae)
        print 'CAE3 built'
        if cae3_model:
            cae3.load(cae3_model)
        sys.stdout.flush()

        if pretrain_layer != 0:
            cae_models = [cae1, cae2, cae3]
            do_pretraining_cae(data_dir=data_dir,
                               models=cae_models,
                               cae_layer=pretrain_layer,
                               max_epoch=100)


    elif finetune or test or get_hidden:
        print 'creating scae...'
        sys.stdout.flush()
        if True not in binary:
            scae = stacked_CAE3d(image_shape=image_shp,
                                 filter_shapes=(filter_shp_1, filter_shp_2, filter_shp_3),
                                 poolsize=(2, 2, 2),
                                 activation_cae=activation_cae,
                                 activation_final=activation_final,
                                 hidden_size=(2000, 500, 200, 20, 3))
        else:
            scae = stacked_CAE3d(image_shape=image_shp,
                                 filter_shapes=(filter_shp_1, filter_shp_2, filter_shp_3),
                                 poolsize=(2, 2, 2),
                                 activation_cae=activation_cae,
                                 activation_final=activation_final,
                                 hidden_size=(2000, 500, 200, 20, 2))

        print 'scae model built'
        sys.stdout.flush()
        if cae1_model:
            scae.load_cae(cae1_model, cae_layer=0)
            pass
        if cae2_model:
            scae.load_cae(cae2_model, cae_layer=1)
            pass
        if cae3_model:
            scae.load_cae(cae3_model, cae_layer=2)
            pass
        sys.stdout.flush()

        if scae_model:
            if True in binary and scae_model[:-25] == 'scae':
                if load_conv:
                    scae.load_conv(scae_model)
                else:
                    scae.load_binary(scae_model)
            else:
                if load_conv:
                    scae.load_conv(scae_model)
                else:
                    scae.load(scae_model)
        pass
        sys.stdout.flush()

        if finetune:
            finetune_scae_crossvalidate(data_dir=data_dir,
                                        model=scae,
                                        binary_classification=binary,
                                        max_epoch=100)
        elif test:
            test_scae_crossvalidate(data_dir=data_dir,
                                    model=scae,
                                    binary_classification=binary)
        elif get_hidden!=0:
            get_hidden_finetuned(dir=data_dir,
                                 model=scae,
                                 layer=get_hidden)


if __name__ == '__main__':
    sys.exit(main())

