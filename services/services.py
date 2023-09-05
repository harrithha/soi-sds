import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


def check_format(df):
    # correct = True
    error_msg = ""
    required_columns = {'kepid': 0, 'tce_plnt_num': 0, 'tce_rogue_flag': 0, 'tce_period': 0,
                        'tce_period_err': 0, 'tce_time0bk': 0, 'tce_time0bk_err': 0, 'tce_impact': 0,
                        'tce_impact_err': 0, 'tce_duration': 0, 'tce_duration_err': 0, 'tce_depth': 0,
                        'tce_depth_err': 0, 'tce_model_snr': 0, 'tce_prad': 0, 'tce_prad_err': 0, 'tce_eqt': 0,
                        'tce_eqt_err': 0, 'tce_insol': 0, 'tce_insol_err': 0, 'tce_steff': 0,
                        'tce_steff_err': 0, 'tce_slogg': 0, 'tce_slogg_err': 0, 'tce_sradius': 0,
                        'tce_sradius_err': 0}

    columns = df.columns

    for i in columns:
        if(i not in required_columns.keys()):
            df.drop(i, inplace=True, axis=1)

        else:
            if(required_columns[i] == 0):
                required_columns[i] = 1
            else:
                df.drop(i, inplace=True, axis=1)

    for i in required_columns:
        if(required_columns[i] == 0):
            error_msg = i + " column is missing"
            return False, error_msg

    types = df.dtypes
    for i in required_columns:
        if(types[i] != 'float' and types[i] != 'int'):
            error_msg = "Data type of " + str(i) + " should be int or float "
            return False, error_msg

    return True, error_msg


def predict(df):

    delete = ["kepid", "tce_plnt_num", "tce_rogue_flag", "tce_insol", "tce_impact", "tce_insol_err", "tce_period_err", "tce_time0bk_err", "tce_impact_err",
              "tce_duration_err", "tce_depth_err", "tce_prad_err", "tce_eqt_err", "tce_eqt_err", "tce_steff_err", "tce_slogg_err", "tce_sradius_err"]
    data_new = df.drop(delete, axis=1)
    data_new.dropna(subset=['tce_period', 'tce_time0bk', 'tce_duration', 'tce_depth', 'tce_model_snr',
                    'tce_prad', 'tce_eqt', 'tce_steff', 'tce_slogg', 'tce_sradius'], inplace=True)

    mean = {
        'tce_period': 2.93291512e+01,
        'tce_time0bk': 1.43612343e+02,
        'tce_duration': 4.80776600e+00,
        'tce_depth': 1.27949972e+04,
        'tce_model_snr': 1.90403268e+02,
        'tce_prad': 8.15391768e+00,
        'tce_eqt': 1.86699887e+03,
        'tce_steff': 6.33844191e+03,
        'tce_slogg': 4.19125150e+00,
        'tce_sradius': 2.43331856e+00
    }

    std_dev = {
        'tce_period': 9.24820052e+01,
        'tce_time0bk': 3.97876643e+01,
        'tce_duration': 4.24394720e+00,
        'tce_depth': 6.92931174e+04,
        'tce_model_snr': 1.06404264e+03,
        'tce_prad': 4.78599773e+01,
        'tce_eqt': 1.28271642e+03,
        'tce_steff': 1.15825662e+03,
        'tce_slogg': 4.87043496e-01,
        'tce_sradius': 8.97215415e+00
    }
    for i in mean:
        data_new[i] -= mean[i]
        data_new[i] /= std_dev[i]

    model = pickle.load(open('./ML/nnc.pkl', "rb"))
    model_type = "nnc"

    # prediction coloumn
    prediction = model.predict(data_new)

    # new df extra coloumn
    df['predicted'] = prediction

    return df, model_type
