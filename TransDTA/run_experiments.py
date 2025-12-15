from __future__ import print_function
import numpy as np
import pywt
import tensorflow as tf
import tensorflow.compat.v1 as tf2
import time
import os
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rn

from matplotlib import gridspec
from sklearn.metrics import average_precision_score
from tensorflow import keras
from keras import backend as K
from keras.layers import Conv1D, SeparableConv1D, Input, Embedding, Dense, Dropout, Activation, \
    Bidirectional, Add, LSTM, GRU, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping
from copy import deepcopy

from arguments import argparser, loggingdef
from datahelper import DataSet
from emetrics import get_rm2

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.LogicalDeviceConfiguration(memory_limit= -1)
    except RuntimeError as e:
        print(e)

mpl.rcParams.update(mpl.rcParamsDefault)

tf2.disable_v2_behavior()
os.environ['PYTHONHASHSEED'] = '0'


np.random.seed(1)
rn.seed(1)


session_conf = tf2.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf2.set_random_seed(0)

sess = tf2.Session(graph=tf2.get_default_graph(), config=session_conf)
K.set_session(sess)

sns.set_theme(style='white')

figdir = "figures/"


# # level：（1 -> {2}；2 -> {2,4}；3 -> {2,4,8}）
# DWA_LEVEL = 3
# # 'none' | 'soft' | 'hard'
# DWA_THRESH_MODE = 'soft'
# DWA_THRESH_ALPHA = 1
# DWA_THRESH_VALUE = 0.10
# DWA_AVG_AGG = True
# 
# def _crop_or_pad_to_len(x_ref, x_to_fit):
#     lx = K.shape(x_ref)[1]; ll = K.shape(x_to_fit)[1]
#     return tf.cond(
#         ll > lx,
#         lambda: x_to_fit[:, :lx, :],
#         lambda: tf.pad(x_to_fit, [[0,0],[0, lx-ll],[0,0]])
#     )
# 
# def DWA_block(x, name="dwa"):
#     with K.name_scope(name):
#         C = K.int_shape(x)[-1]
#         scales = [2 ** i for i in range(1, max(1, int(DWA_LEVEL)) + 1)]
# 
#         low_list, high_list = [], []
#         for s in scales:
#             low = AveragePooling1D(pool_size=s, strides=s, padding='same', name=f'{name}_avg{s}')(x)
#             low_up = UpSampling1D(size=s, name=f'{name}_up{s}')(low)
#             low_up = Lambda(lambda t: _crop_or_pad_to_len(x, t), name=f'{name}_crop{s}')(low_up)
# 
#             high = keras.layers.Subtract(name=f'{name}_high{s}')([x, low_up])
# 
#             if DWA_THRESH_MODE != 'none':
#                 # sigma：用 batch 内 std 近似噪声尺度（不引入 tf-probability）
#                 sigma = Lambda(lambda t: K.std(t, axis=[1,2], keepdims=True), name=f'{name}_sigma{s}')(high)
#                 if DWA_THRESH_VALUE is None:
#                     tau = Lambda(lambda t: DWA_THRESH_ALPHA * t, name=f'{name}_tau{s}')(sigma)
#                 else:
#                     tau = Lambda(lambda t: K.ones_like(t) * float(DWA_THRESH_VALUE), name=f'{name}_taufix{s}')(sigma)
# 
#                 if DWA_THRESH_MODE == 'soft':
#                     # soft-shrink: sign(x) * max(|x|-tau, 0)
#                     high = Lambda(lambda t: K.sign(t) * K.maximum(K.abs(t) - tau, 0.0),
#                                   name=f'{name}_soft{s}')(high)
#                 elif DWA_THRESH_MODE == 'hard':
#                     # hard-shrink: x * 1{|x| > tau}
#                     high = Lambda(lambda t: t * K.cast(K.greater(K.abs(t), tau), K.floatx()),
#                                   name=f'{name}_hard{s}')(high)
# 
#             low_list.append(low_up); high_list.append(high)
# 
#         if len(low_list) == 1:
#             low_agg = low_list[0]; high_agg = high_list[0]
#         else:
#             low_agg  = keras.layers.Add(name=f'{name}_low_add')(low_list)
#             high_agg = keras.layers.Add(name=f'{name}_high_add')(high_list)
#             if DWA_AVG_AGG:
#                 inv_n = 1.0 / float(len(low_list))
#                 low_agg  = Lambda(lambda t: t * inv_n,  name=f'{name}_low_avg')(low_agg)
#                 high_agg = Lambda(lambda t: t * inv_n,  name=f'{name}_high_avg')(high_agg)
# 
#         gap_low  = GlobalAveragePooling1D(name=f'{name}_gap_low')(low_agg)
#         gap_high = GlobalAveragePooling1D(name=f'{name}_gap_high')(high_agg)
#         gaps = Concatenate(name=f'{name}_gaps')([gap_low, gap_high])  
# 
#         att = Dense(32, activation='relu', name=f'{name}_fc1')(gaps)
#         att = Dense(2, name=f'{name}_fc2')(att)
#         att = Activation('softmax', name=f'{name}_softmax')(att)  # [B, 2]
#         w_low  = Lambda(lambda t: K.reshape(t[:, 0], (-1,1,1)), name=f'{name}_w_low')(att)
#         w_high = Lambda(lambda t: K.reshape(t[:, 1], (-1,1,1)), name=f'{name}_w_high')(att)
# 
#         low_w  = Multiply(name=f'{name}_mul_low')([low_agg,  w_low])
#         high_w = Multiply(name=f'{name}_mul_high')([high_agg, w_high])
# 
#         y = keras.layers.Add(name=f'{name}_sum')([low_w, high_w])  
# 
#         y = Conv1D(filters=C, kernel_size=3, padding='same', activation='relu', name=f'{name}_smooth')(y)
#         return y


def wavelet_transform(data):
    wavelet = 'db4'
    level = 3

    coeffs = pywt.wavedec(data, wavelet, level=level)

    threshold = 0.1
    coeffs_thresholded = [pywt.threshold(c, threshold) for c in coeffs]

    reconstructed_data = pywt.waverec(coeffs_thresholded, wavelet)

    return reconstructed_data


def build_combined_categorical1(FLAGS, NUM_FILTERS, FILTER_LENGTH1):

    global proteinFeatures
    fpath = FLAGS.dataset_path
    if fpath=='data/davis/':
        proteinFeatures = 442
    elif fpath=='data/kiba/':
        proteinFeatures = 229


    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='float32')
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)

    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(encode_smiles)
    encode_smiles = SeparableConv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu',
                                    padding='valid', strides=1)(encode_smiles)
    encode_smiles = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(encode_smiles)
#  encode_smiles = DWA_block(encode_smiles, name='DWA_smiles')
    residual_x = encode_smiles
    x = Bidirectional(GRU(units=128, return_sequences=True))(encode_smiles)
    residual_x = Dense(K.int_shape(x)[-1], activation=None)(residual_x)
    residual_x = keras.layers.BatchNormalization()(residual_x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, residual_x])
    x = Bidirectional(GRU(units=NUM_FILTERS, return_sequences=False))(x)
    encode_smiles = x  # 更新encode_smiles为加入残差后的输出

    XTinput = Input(shape=(proteinFeatures,), dtype='float32')
    protein_reshaped = Reshape((proteinFeatures, 1))(XTinput)
    x_protein = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(protein_reshaped)
    x_protein = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x_protein)
    x_protein = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x_protein)
    x_protein = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x_protein)
#  x_protein = DWA_block(x_protein, name='DWA_protein')
    residual_protein = keras.layers.BatchNormalization()(x_protein)
    x_protein = keras.layers.BatchNormalization()(x_protein)
    x_protein = Add()([x_protein, residual_protein])
    x_protein = Activation('relu')(x_protein)
    x_protein = Bidirectional(LSTM(units=64, return_sequences=False))(x_protein)
    residual_fc = Dense(128, activation=None)(XTinput)
    x_protein = Add()([x_protein, residual_fc])
    x_protein = Activation('relu')(x_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, x_protein], axis=-1)
    fc_input = encode_interaction
    fc_residual = Dense(512, activation=None)(fc_input)
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.2)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.2)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)
    FC_out = keras.layers.Add()([FC2, fc_residual])
    predictions = Dense(1, kernel_initializer='normal')(FC_out)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    opt = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)

    interactionModel.compile(optimizer=opt, loss='mean_squared_error',
                             metrics=[cindex_score])

    print(interactionModel.summary())
    return interactionModel

def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, runmethod, FLAGS, dataset):

    test_set, outer_train_sets = dataset.read_sets(FLAGS)

    foldinds = len(outer_train_sets)

    test_sets = []
    val_sets = []
    train_sets = []

    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT, Y,
                                                                                                          label_row_inds,
                                                                                                          label_col_inds,
                                                                                                          runmethod,
                                                                                                          FLAGS,
                                                                                                          train_sets,
                                                                                                          val_sets)

    print("Test Set len", str(len(test_set)))
    print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, Y, label_row_inds,
                                                                                         label_col_inds,
                                                                                         runmethod, FLAGS,
                                                                                         train_sets, test_sets)

    loggingdef("---FINAL RESULTS-----", FLAGS)
    loggingdef("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    loggingdef("Test Performance CI", FLAGS)
    loggingdef(testperfs, FLAGS)
    loggingdef("Test Performance MSE", FLAGS)
    loggingdef(testloss, FLAGS)

    return avgperf, avgloss, teststd


def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, runmethod, FLAGS, labeled_sets, val_sets):

    paramset1 = FLAGS.num_windows
    paramset2 = FLAGS.smi_window_lengths
    epoch = FLAGS.num_epoch
    batchsz = FLAGS.batch_size

    loggingdef("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]

        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        pointer = 0

        for param1ind in range(len(paramset1)):
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):
                param2value = paramset2[param2ind]

                gridmodel = runmethod(FLAGS, param1value, param2value)
                es = EarlyStopping(monitor='val_cindex_score', mode='max', verbose=1, patience=200, restore_best_weights=True, )
                gridres = gridmodel.fit(([np.array(train_drugs), np.array(train_prots)]), np.array(train_Y), batch_size=batchsz, epochs=epoch,
                                        validation_data=(
                                            ([np.array(val_drugs), np.array(val_prots)]), np.array(val_Y)),
                                        shuffle=False, callbacks=[es])

                predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots)])

                lst = gridres.history['val_cindex_score']
                lst2 = gridres.history['val_loss']
                rperf = max(lst)
                index = lst.index(rperf)
                loss = lst2[index]

                rm2 = get_rm2(val_Y, predicted_labels)

                predicted_labels = predicted_labels.tolist()
                labels = []
                for i in range(len(predicted_labels)):
                    labels.append(predicted_labels[i][0])

                fpath = FLAGS.dataset_path
                if fpath=='data/kiba/':
                    thresh = 12.1
                else:
                    thresh = 7

                temp = []
                for i in range(len(val_Y)):
                    if (val_Y[i] > thresh):
                        temp.append(1)
                    else:
                        temp.append(0)

                aupr = average_precision_score(temp, labels)

                loggingdef(
                    "CI = %f, MSE = %f, aupr = %f , r2m = %f" %
                    (rperf, loss, aupr, rm2), FLAGS)

                df = pd.DataFrame(list(zip(labels, val_Y)),
                                  columns=['Prediction Result', 'Measured Result'])
                g = sns.relplot(data=df, x="Prediction Result", y="Measured Result", color="#e9967a", s=20)
                g.fig.set_size_inches(6.6, 5.5)

                x_data = df["Prediction Result"].tolist()
                y_data = df["Measured Result"].tolist()

                gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

                ax_main = plt.subplot(gs[1, 0])

                ax_main.scatter(x_data, y_data, s=5, c='#6495ed', edgecolors=(1, 0, 0, 0), linewidths=0.01,
                                label='DAVIS Data Points')

                ax_main.legend(fontsize='6')

                ax_main.set_xlim(4, 11)
                ax_main.set_ylim(4, 11)

                fit = np.polyfit(x_data, y_data, 1)
                m = fit[0]
                b = fit[1]

                x_fit = np.array([min(x_data), max(x_data)])
                y_fit = m * x_fit + b
                ax_main.plot(x_fit, y_fit, 'red', label='Line of best fit', linewidth=1)

                ax_main.legend(fontsize='6')

                ax_main.set_title("DAVIS Dataset")

                x_vals = np.array([4, 11])

                y_vals = x_vals
                ax_main.plot(x_vals, y_vals, 'k--', label='y = x', linewidth=1)

                plt.style.use('default')
                ax_main.legend(fontsize='6')

                ax_main.set_xlabel('Prediction Result')
                ax_main.set_ylabel('Measured Result')


                ax_main.set_xlim(4, 11)
                ax_main.set_ylim(4, 11)

                plt.tight_layout()

                figname = "scatter_b" + str(param1ind) + "_e" + str(param2ind) + "_" + str(foldind) + "_" + str(time.time())
                plt.savefig("figures/" + figname + ".tiff", bbox_inches='tight', pad_inches=0.5, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
                plt.close()

                plotLoss(gridres, param1ind, param2ind, foldind)

                all_predictions[pointer][foldind] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                all_losses[pointer][foldind] = loss

                pointer += 1

    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []

    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):

                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf2.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)
    result = tf.where(tf.equal(g, 0), 0.0, g / f)

    return result


def plotLoss(history, batchind, epochind, foldind):
    figname = "b" + str(batchind) + "_e" + str(epochind) + "_" + str(foldind) + "_" + str(
        time.time())

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.savefig("figures/" + figname + ".png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()
    plt.figure()
    plt.ylabel('c_index(CI)')
    plt.xlabel('Epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['train_c_index', 'val_c_index'], loc='lower right')
    plt.savefig("figures/" + figname + "_acc.png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, deepmethod, foldcount=6):

    dataset = DataSet(fpath=FLAGS.dataset_path,
                      setting_no=FLAGS.problem_type,
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)

    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    XD_wavelet = []
    for drug_data in XD:
        reconstructed_drug_data = wavelet_transform(drug_data)
        XD_wavelet.append(reconstructed_drug_data)
    XD_wavelet = np.array(XD_wavelet)

    XT_wavelet = []
    for target_data in XT:
        reconstructed_drug_data = wavelet_transform(target_data)
        XT_wavelet.append(reconstructed_drug_data)
    XT_wavelet = np.array(XT_wavelet)

    drugcount = XD_wavelet.shape[0]
    print(drugcount)
    targetcount = XT_wavelet.shape[0]
    print(targetcount)
    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)

    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                    deepmethod, FLAGS, dataset)

    loggingdef("Setting " + str(FLAGS.problem_type), FLAGS)
    loggingdef("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    deepmethod = build_combined_categorical1
    experiment(FLAGS, deepmethod)


if __name__ == "__main__":

    FLAGS = argparser()

    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    loggingdef(str(FLAGS), FLAGS)

    run_regression(FLAGS)
