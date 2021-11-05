import os, sys

import sklearn.datasets as datasets

from pandas import DataFrame
import pandas as pd
from tabulate import tabulate
import sklearn.preprocessing as preprocessing
# from fancyimpute import KNN, SoftImpute, BiScaler, IterativeSVD, NuclearNormMinimization
import datetime
import numpy as np
now = datetime.datetime.now()

dataDir = os.path.join(os.getcwd(), 'data')  # default unless specified otherwise

class Data(object): 
    label = 'label'
    features = []

def get_diabetes_data(): 
    # https://www.kaggle.com/uciml/pima-indians-diabetes-database#diabetes.csv
    
    fpath = 'data/diabetes.csv'
    # diabetes dataset
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset  
    pima = pd.read_csv(fpath, header=1, names=col_names)  # header=None
    print("> columns: {}".format(col_names))
    
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose', 'bp', 'pedigree']
    X = pima[feature_cols] # Features
    y = pima.label # Target variable
    
    print("> data layout:\n{}\n".format(pima.head()))
    
    return (X, y, feature_cols) 



def preprocess_data(df):
    """
    Encode categorical and ordinal features. 

    References
    ----------
    1. dealing with categorical data

       a. https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63

       b. one-hot encoding

          https://www.ritchieng.com/machinelearning-one-hot-encoding/ 

    2. One-hot encoding in sklearn 

        a. The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features.
        b. The output will be a sparse matrix where each column corresponds to one possible value of one feature.
        c. It is assumed that input features take on values in the range [0, n_values).
        d. This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # le = LabelEncoder()

    # # first we need to know which columns are ordinal, categorical 
    # col = ''
    # num_labels = le.fit_transform(df[col])
    # mappings = {index: label for index, label in enumerate(le.classes_)} 

    # limit to categorical data using df.select_dtypes()
    df = df.select_dtypes(include=[object])
    print(df.head(3))
    
    cols = df.columns  # categorical data candidates  <<< edit here

    le = preprocessing.LabelEncoder()

    # 2/3. FIT AND TRANSFORM
    # use df.apply() to apply le.fit_transform to all columns
    df2 = df.apply(df.fit_transform)
    print(df2.head())
    # ... now all the categorical variables have numerical values

    # INSTANTIATE
    enc = preprocessing.OneHotEncoder()

    # FIT
    enc.fit(df2)

    # 3. Transform
    onehotlabels = enc.transform(df2).toarray()
    

    return

def extracting_pollutant(pollutant_cat_df, yr_col_name, poullutant_yr_col_name):
    yr_col = pollutant_cat_df[yr_col_name]
    for idx, yr in enumerate(yr_col):
        pollutant_cat_df.loc[idx, poullutant_yr_col_name] = pollutant_cat_df.loc[idx, yr]
    return pollutant_cat_df

def load_merge(vars_matrix='exposures-4yrs.csv',
               # label_matrix='nasal_biomarker_asthma1019.csv',
               label_matrix='nasal_biomarker_asthma1019.csv',
               avg_income_matrix='Zip_avg_income.csv',
               no2_matrix='NO2_Zipcode.csv',
               pm25_matrix = 'PM2.5_Zipcode.csv',
               output_matrix='exposures-4yrs-asthma.csv',
               backup=False, save=True, verify=False, sep=',',
               tImputation = True, time_window = 0,
               outcome_limited_with_asthma=False,
               binary_outcome=True):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from shutil import copyfile
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
 
    null_threshold = 0.8
    generic_key = 'ID'
    # tImputation = True

    # A. load variables 
    primary_key = 'Study.ID'
    droplist_vars = ['DoB' , 'Zipcode', ]  # 'Zipcode',  'Gender'

    # ... DoB needs further processing 
    categorical_vars = ['Zipcode',  'Gender']

    fpath_vars = os.path.join(dataDir, vars_matrix)
    assert os.path.exists(fpath_vars), "(load_merge) Invalid path to the variable matrix"
    exposure_df = pd.read_csv(fpath_vars, header=0, sep=sep)
    print("(load_merge) dim(dfv): {} | vars:\n... {}\n".format(exposure_df.shape, exposure_df.columns.values))

    # backup 
    # if backup: 
    #     vars_matrix_bk = "{}.bk".format(vars_matrix)
    #     fpath_vars_bk = os.path.join(dataDir, vars_matrix_bk)
    #     copyfile(fpath_vars, fpath_vars_bk)
    #     print("... copied {} to {} as a backup".format(vars_matrix, vars_matrix_bk))

    # transform
    msg = ''
    le = LabelEncoder()
    dfvc = exposure_df[categorical_vars]
    msg += "... original dfvc:\n"
    msg += tabulate(dfvc.head(5), headers='keys', tablefmt='psql') + '\n'

    dfvc2 = dfvc.apply(le.fit_transform)
    msg += "... transformed dfvc:\n"
    msg += tabulate(dfvc2.head(5), headers='keys', tablefmt='psql')
    exposure_df[categorical_vars] = dfvc2[categorical_vars]

    # mapping
    if verify: 
        for cvar in categorical_vars:
            mapping = dict(zip(dfvc[cvar], dfvc2[cvar]))
            cols = [cvar, '{}_numeric'.format(cvar)]
            dfmap = DataFrame(mapping, columns=cols)
            msg += tabulate(dfmap.head(10), headers='keys', tablefmt='psql') + '\n'

            uvals = dfvc[cvar].unique()
            for uval in uvals: 
                print("... {} | {} -> {}".format(cvar, uval, mapping[uval]))

    # drop columns 
    exposure_df = exposure_df.drop(droplist_vars, axis=1)

    # rename ID column
    exposure_df = exposure_df.rename(columns={primary_key: generic_key})
    msg += "... full transformed data:\n"
    msg += tabulate(exposure_df.head(3), headers='keys', tablefmt='psql') + '\n'

    msg += "... final dim(vars_matrix): {} | vars:\n... {}\n".format(exposure_df.shape, exposure_df.columns.values)
    N0 = exposure_df.shape[0]
    assert generic_key in exposure_df.columns
    print(msg)




    # remove rows with all NaN? 
    # null_threshold = 0.8
    msg += "... dropping rows with predominently null values (thresh={})\n".format(null_threshold)
    Nu, Ni = exposure_df.shape

    # if tDropNA: 
    Nth = int(Ni * null_threshold)  # Keep only the rows with at least Nth non-NA values.
    msg += "...... keeping only rows with >= {} non-NA values\n".format(Nth)
    # dfv.dropna(thresh=Nth, axis=0, inplace=True) # how='all',

    # Richard: NA is dropped here
    exposure_df.dropna(how='any', inplace=True)
    # print('exposure_df #cols:', exposure_df.)
    # exposure_df.dropna(thresh=int(0.8*20), axis=0, inplace=True)
    # exposure_df.dropna(how='any', axis=0, inplace=True)
    # msg += "> after dropping rows with predominently null, n_rows: {} -> {} | dim(dfv): {}\n".format(N0, exposure_df.shape[0], exposure_df.shape)

    if tImputation:
        msg += "... applying multivarite data imputation\n"
        # imp = IterativeImputer(max_iter=60, random_state=0)
        imp = IterativeSVD()
        # imp = SoftImpute()
        # imp = NuclearNormMinimization()
        drop_list = [generic_key, 'Gender']
        dfx = exposure_df.drop(drop_list, axis=1)
        features = dfx.columns
        X = dfx.values
        print(X.shape)
        y = exposure_df[drop_list].values
        print(y)

        X = imp.fit_transform(X)
        exposure_df = DataFrame(X, columns=features)
        dropped_df = DataFrame(y, columns=drop_list)
        exposure_df[drop_list] = dropped_df
        # assert exposure_df.shape[0] == exposure_df.dropna(how='any').shape[0], "dfv.shape[0]: {}, dfv.dropna.shape[0]: {}".format(exposure_df.shape[0], exposure_df.dropna(how='any').shape[0])

    print(exposure_df.shape)
    ####################################################
    # B. load labels (from yet another file)
    # Richard: TODO: Adding confounding factors here to load
    primary_key = 'Study ID'
    age = "Today's age: (years)"
    label = 'Has a doctor ever diagnosed you with asthma?'
    # outcome = 'ACT score'
    # outcome = 'Have you ever needed to go to the emergency department for asthma?'
    # outcome = 'Have you ever needed to be hospitalized over night for asthma?'
    # outcome = 'Do you regularly use an asthma medication (one that has been prescribed by a physician for regular use)?'
    # outcome = 'In the past 6 months, have you had regular (2/week) asthma symptoms?  '
    # outcome = 'In the past 6 mos, have you been on a daily controller asthma medication (e.g. Qvar, Flovent, Advair, Symbicort, Singulair)?'
    # outcome = 'At what age (in years) were you diagnosed with asthma? '
    # outcome = 'How many times in the past year have you gone to the emergency department for asthma? '
    # outcome = 'How many times in the past year have you been hospitalized over night?'
    outcome = 'During the past week, how many days have you had asthma symptoms (e.g. shortness of breath, wheezing, chest tightness)?'



    gender = 'Gender'
    race = 'Race'

    # fever_sym = "Do you currently have hay fever symptoms (runny, stuffy nose accompanied by sneezing and itching when you did not have a cold or flu--sometimes called 'allergies')?"

    # only take these columns
    if outcome_limited_with_asthma:
        labelmap = {
            outcome: 'label',
            age: 'age',
            gender: 'gender',
            # race: 'race',
            # fever_sym: 'fever_symptom'
        }
        col_names = [primary_key, label, age, gender, outcome
                     # race
                     ]
        categorical_vars = [
            gender,
            # race
        ]
        if binary_outcome:
            categorical_vars.append(outcome)
    else:
        labelmap = {
            label: 'label',
            age: 'age',
            gender: 'gender',
            # race: 'race',
            # fever_sym: 'fever_symptom'
        }
        col_names = [primary_key, label, age, gender
                     # race
                     ]
        categorical_vars = [
            label,
            gender,
            # race
        ]

    # ... use the simpler new variable names?

    fpath_labels = os.path.join(dataDir, label_matrix)
    assert os.path.exists(fpath_labels), "(load_merge) Invalid path to the label matrix"
    asthma_df = pd.read_csv(fpath_labels, header=0, sep=sep)

    # Race need to be one-hot encoded
    race_df = pd.read_csv('./data/RaceEthnicity_YC_05042021.csv', header=0, sep=sep)
    one_hot_encoded_cols = ["race/ethnicity"]

    one_hot_encoded_df = pd.get_dummies(race_df[one_hot_encoded_cols], prefix=one_hot_encoded_cols)



    zip_code_df = asthma_df[['What is your zip code?']]
    birth_yr_zip_df = pd.read_csv(fpath_labels, header=0, sep=sep)[[primary_key,
                                                                    "Date of Birth",
                                                                    'What is your zip code?']]

    birth_yr_zip_df.dropna(how='any', inplace=True)
    birth_yr_zip_df["birth_year"] = pd.to_datetime(birth_yr_zip_df["Date of Birth"], format="%m/%d/%y").dt.year
    birth_yr_zip_df['zip'] = birth_yr_zip_df['What is your zip code?'].apply(int)

    list_pollutant = ['EC', 'OC', 'SO4', 'NH4', 'Nit', 'NO2', 'PM2.5']
    pollutant_dict = dict()
    for pol in list_pollutant:
        pollutant_dict[pol] = pd.read_csv(os.path.join(dataDir, '{}_Zipcode.csv'.format(pol)))
    # no2_df = pd.read_csv(os.path.join(dataDir, no2_matrix))
    # pm25_df = pd.read_csv(os.path.join(dataDir, pm25_matrix))
    # pollutant_dict = {'NO2': no2_df,
    #                   'PM2.5': pm25_df,
    #                 }
    if type(time_window) is range:
        extracting_range = time_window
        time_pt_filename = "y{}to{}".format(time_window[0], time_window[-1])
    else:
        time_pt_filename = "y{}".format(time_window)
        extracting_range = [time_window]
    # extracting_range = range(-1, 5)
    birth_yr_zip_df['y_nan'] = np.nan


    pollutant_cat_df = birth_yr_zip_df.copy()
    pollutant_cat_df.rename(columns={primary_key: 'ID'}, inplace=True)
    # time_windows_list = [-1, 0, range(-1, 1),
    #                 range(2), range(3), range(4), range(5),
    #                 range(-1, 2), range(-1, 3), range(-1, 4), range(-1, 5)]


    # for idx, pat in birth_yr_zip_df:
    pollutant_year_col = []
    for pollutant_str, pollutant_df in pollutant_dict.items():

        pollutant_df.rename(columns={'Zipcode':'zip'}, inplace=True)
        # pollutant_df['zip'] = pollutant_df['SiteData_point.zipcode'].apply(int)
        pollutant_df.drop_duplicates('zip', inplace=True)
        # pollutant_df['y_nan'] = np.nan
        pollutant_cols = pollutant_df.columns
        drop_col_list = list(pollutant_df.columns)
        drop_col_list.remove('zip')
        pollutant_cat_df = pollutant_cat_df.merge(pollutant_df, on='zip', how='outer')
        # print(pollutant_cat_df.columns)
        for extracting_yr in extracting_range:
            current_y_str = "y{}".format(extracting_yr)
            pollutant_cat_df[current_y_str] = "y" + (pollutant_cat_df["birth_year"] + extracting_yr).apply(
                str)
            pollutant_cat_df[current_y_str+'_filtered'] = pollutant_cat_df[current_y_str].apply(lambda x: x if x in pollutant_cols else 'y_nan')
            pollutant_cat_df[pollutant_str+current_y_str] = 0
            pollutant_cat_df = extracting_pollutant(pollutant_cat_df, current_y_str+'_filtered', pollutant_str + current_y_str)

            # pollutant_cat_df[pollutant_str + current_y_str] = pollutant_cat_df[current_y_str+'_filtered'].apply(lambda x: pollutant_cat_df[x])
            drop_col_list.append(current_y_str)
            drop_col_list.append(current_y_str+'_filtered')
            pollutant_year_col.append(pollutant_str+current_y_str)

        pollutant_cat_df.drop(drop_col_list, axis=1, inplace=True)
        # print(pollutant_cat_df['ID'])

    pollutant_drop_list = ['y_nan', 'Date of Birth', 'What is your zip code?', 'zip']
    pollutant_cat_df.drop(pollutant_drop_list, axis=1, inplace=True)
                # pollutant_cat_df.loc[pollutant_time_window_colname]




    # exposure_df.dropna(thresh=int(0.8*20), axis=0, inplace=True)
    # exposure_df.dropna(how='any', axis=0, inplace=True)
    # msg += "> after dropping rows with predominently null, n_rows: {} -> {} | dim(dfv): {}\n".format(N0, exposure_df.shape[0], exposure_df.shape)
    print('Remaining Columns before imputation', pollutant_cat_df.columns)
    if tImputation:
        drop_list = [generic_key]
        no_cols_for_imputating = len(pollutant_cat_df.columns)-len(drop_list)
        pollutant_cat_df.dropna(thresh=int(0.8*no_cols_for_imputating),
                                axis=0, inplace=True)
        # imp = IterativeImputer(max_iter=60, random_state=0)
        imp = IterativeSVD()
        # imp = SoftImpute()
        # imp = NuclearNormMinimization()

        dfx = pollutant_cat_df.drop(drop_list, axis=1)
        features = dfx.columns
        X = dfx.values
        print(X.shape)
        y = pollutant_cat_df[drop_list].values
        print(y)

        X = imp.fit_transform(X)
        pollutant_cat_df = DataFrame(X, columns=features)
        dropped_df = DataFrame(y, columns=drop_list)
        pollutant_cat_df[drop_list] = dropped_df
    else:
        pollutant_cat_df.dropna(how='any', axis=0, inplace=True)


    for pollutant_str, pollutant_df in pollutant_dict.items():
        # for time_window in time_windows_list:
        if type(time_window) is range:
            time_pt_taken = len(time_window)
            pollutant_time_window_colname = "{}y{}to{}".format(pollutant_str, time_window[0], time_window[-1])
            pollutant_cat_df[pollutant_time_window_colname] = 0
            for time_pt in time_window:
                time_pt_col = "{}y{}".format(pollutant_str, time_pt)
                pollutant_cat_df.loc[:, pollutant_time_window_colname] += pollutant_cat_df[
                                                                              time_pt_col] / time_pt_taken
        else:
            pollutant_time_window_colname = "{}y{}".format(pollutant_str, time_window)
            pollutant_year_col.remove(pollutant_time_window_colname)

    pollutant_cat_df.drop(pollutant_year_col, axis=1, inplace=True)

    asthma_df = asthma_df[col_names]
    if outcome_limited_with_asthma:
        asthma_df = asthma_df[asthma_df[label]=='Yes']
        asthma_df.drop(columns=[label], inplace=True)
    asthma_df.dropna(how='any', axis=0, inplace=True)
    # assert asthma_df.shape[1] == len(col_names)
    print(tabulate(asthma_df.head(5), headers='keys', tablefmt='psql'))

    # asthma_df.loc[asthma_df[outcome] < 5, outcome] = 0
    # asthma_df.loc[asthma_df[outcome] >= 5, outcome] = 1

    # rename
    # dfl = dfl.rename(columns={primary_key: generic_key, label: labelmap[label]})
    # print("(load_merge) dim(dfl): {} | cols:\n... {}\n".format(dfl.shape, dfl.columns.values))

    # transform
    msg = ''
    le = LabelEncoder()
    dflc = asthma_df[categorical_vars]
    msg += "... original dflc:\n"
    msg += tabulate(dflc.head(5), headers='keys', tablefmt='psql') + '\n'
    print(msg)

    # dflc2 = dflc.apply(le.fit_transform)  # log: '<' not supported between instances of 'str' and 'float'

    # transform column by column
    cols_encoded = {cvar: {} for cvar in categorical_vars}
    for cvar in categorical_vars:
        cvar_encoded = '{}_numeric'.format(cvar)
        asthma_df[cvar] = le.fit_transform(asthma_df[cvar].astype(str))

        # keep track of mapping

        cols_encoded[cvar] = {label: index for index, label in enumerate(le.classes_)}  # encoded value -> name

    # dfl[categorical_vars] = dflc2[categorical_vars]
    print(cols_encoded)
    # mapping
    if verify:
        msg += "> verifying the (value) encoding ...\n"
        for cvar in categorical_vars:
            mapping = cols_encoded[cvar]
            for label, index in mapping.items():
                print("... {} | {} -> {}".format(cvar, label, index))

                # rename
    colmap = {primary_key: generic_key}
    colmap.update(labelmap)
    msg += "> renaming columns via {}\n".format(colmap)
    asthma_df = asthma_df.rename(columns=colmap)
    msg += "... transformed dflc:\n"
    msg += tabulate(asthma_df.head(5), headers='keys', tablefmt='psql') + '\n'
    ##################################################
    # Richard: add to merge with avg income and race here
    zip_income_df = pd.read_csv(os.path.join(dataDir, avg_income_matrix))
    # zip_income_df = zip_income_df[['zip', 'Avg_inc']]
    zip_inc_mapping = dict(zip_income_df[['zip', 'Avg_Inc']].values)


    asthma_df['avg_income'] = zip_code_df['What is your zip code?'].map(zip_inc_mapping)

    asthma_df = pd.concat([asthma_df, one_hot_encoded_df], axis=1)
    ###################################################
    print(exposure_df[generic_key])

    # exposure_df = pd.merge(exposure_df, asthma_df, on=generic_key, how='inner')
    # exposure_df = pd.merge(exposure_df, pollutant_cat_df, on=generic_key, how='inner')
    exposure_df = pd.merge(asthma_df, pollutant_cat_df, on=generic_key, how='inner')
    exposure_df.dropna(how='any', inplace=True)
    print(exposure_df.shape)


    # merge
    msg += "> merging variable matrix and label matrix ...\n"

    # finally drop ID columns so that only explanatory and response variables remain
    # exposure_df = exposure_df.drop([generic_key, 'birth_year'], axis=1)
    exposure_df = exposure_df.drop(['birth_year'], axis=1)

    msg += "... final dim(vars_matrix) after including labels: {} | vars:\n... {}\n".format(exposure_df.shape,
                                                                                            exposure_df.columns.values)
    # assert exposure_df.shape[0] == N0, "Some IDs do not have data? prior to join n={}, after join n={}".format(N0,
    #                                                                                                            exposure_df.shape[
    #                                                                                                                0])

    # example_cols = ['Gender', 'ECy1', 'SO4y1', 'NH4y1', 'Nity1', 'OCy1', 'label', ]
    # msg += tabulate(exposure_df[example_cols].head(10), headers='keys', tablefmt='psql') + '\n'

    # msg += "... data after dropping NAs and/or imputation:\n"
    # example_cols = ['Gender', 'ECy1', 'SO4y1', 'NH4y1', 'Nity1', 'OCy1', 'label', ]
    # msg += tabulate(exposure_df[example_cols].head(10), headers='keys', tablefmt='psql') + '\n'

    # print(msg)






    if save: 
        output_path = os.path.join(dataDir, output_matrix.format('_'+time_pt_filename))
        exposure_df.to_csv(output_path, sep=sep, index=False, header=True)
        print('(load_merge) Saved output dataframe to: {}'.format(output_path))



    return exposure_df


def load_merge_income_only(
               # label_matrix='nasal_biomarker_asthma1019.csv',
               label_matrix='nasal_biomarker_asthma1019.csv',
               avg_income_matrix='Zip_avg_income.csv',
               no2_matrix='NO2_Zipcode.csv',
               pm25_matrix='PM2.5_Zipcode.csv',
               output_matrix='exposures-4yrs-asthma.csv',
               backup=False, save=True, verify=False, sep=',',
               tImputation=True, time_window=0,
               outcome_limited_with_asthma=False,
               binary_outcome=True):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from shutil import copyfile
    from sklearn.experimental import enable_iterative_imputer
    # from sklearn.impute import IterativeImputer

    null_threshold = 0.8
    generic_key = 'ID'
    # tImputation = True

    # A. load variables
    primary_key = 'Study.ID'
    droplist_vars = ['DoB', 'Zipcode', ]  # 'Zipcode',  'Gender'

    # ... DoB needs further processing
    categorical_vars = ['Zipcode', 'Gender']

    # fpath_vars = os.path.join(dataDir, vars_matrix)
    # assert os.path.exists(fpath_vars), "(load_merge) Invalid path to the variable matrix"
    # exposure_df = pd.read_csv(fpath_vars, header=0, sep=sep)
    # print("(load_merge) dim(dfv): {} | vars:\n... {}\n".format(exposure_df.shape, exposure_df.columns.values))

    # backup
    # if backup:
    #     vars_matrix_bk = "{}.bk".format(vars_matrix)
    #     fpath_vars_bk = os.path.join(dataDir, vars_matrix_bk)
    #     copyfile(fpath_vars, fpath_vars_bk)
    #     print("... copied {} to {} as a backup".format(vars_matrix, vars_matrix_bk))

    # transform
    # msg = ''
    # le = LabelEncoder()
    # dfvc = exposure_df[categorical_vars]
    # msg += "... original dfvc:\n"
    # msg += tabulate(dfvc.head(5), headers='keys', tablefmt='psql') + '\n'
    #
    # dfvc2 = dfvc.apply(le.fit_transform)
    # msg += "... transformed dfvc:\n"
    # msg += tabulate(dfvc2.head(5), headers='keys', tablefmt='psql')
    # exposure_df[categorical_vars] = dfvc2[categorical_vars]

    # mapping
    # if verify:
    #     for cvar in categorical_vars:
    #         mapping = dict(zip(dfvc[cvar], dfvc2[cvar]))
    #         cols = [cvar, '{}_numeric'.format(cvar)]
    #         dfmap = DataFrame(mapping, columns=cols)
    #         msg += tabulate(dfmap.head(10), headers='keys', tablefmt='psql') + '\n'
    #
    #         uvals = dfvc[cvar].unique()
    #         for uval in uvals:
    #             print("... {} | {} -> {}".format(cvar, uval, mapping[uval]))
    #
    # # drop columns
    # exposure_df = exposure_df.drop(droplist_vars, axis=1)
    #
    # # rename ID column
    # exposure_df = exposure_df.rename(columns={primary_key: generic_key})
    # msg += "... full transformed data:\n"
    # msg += tabulate(exposure_df.head(3), headers='keys', tablefmt='psql') + '\n'
    #
    # msg += "... final dim(vars_matrix): {} | vars:\n... {}\n".format(exposure_df.shape, exposure_df.columns.values)
    # N0 = exposure_df.shape[0]
    # assert generic_key in exposure_df.columns
    # print(msg)
    #
    # # remove rows with all NaN?
    # # null_threshold = 0.8
    # msg += "... dropping rows with predominently null values (thresh={})\n".format(null_threshold)
    # Nu, Ni = exposure_df.shape
    #
    # # if tDropNA:
    # Nth = int(Ni * null_threshold)  # Keep only the rows with at least Nth non-NA values.
    # msg += "...... keeping only rows with >= {} non-NA values\n".format(Nth)
    # # dfv.dropna(thresh=Nth, axis=0, inplace=True) # how='all',
    #
    # # Richard: NA is dropped here
    # exposure_df.dropna(how='any', inplace=True)
    # # print('exposure_df #cols:', exposure_df.)
    # # exposure_df.dropna(thresh=int(0.8*20), axis=0, inplace=True)
    # # exposure_df.dropna(how='any', axis=0, inplace=True)
    # # msg += "> after dropping rows with predominently null, n_rows: {} -> {} | dim(dfv): {}\n".format(N0, exposure_df.shape[0], exposure_df.shape)
    #
    # if tImputation:
    #     msg += "... applying multivarite data imputation\n"
    #     # imp = IterativeImputer(max_iter=60, random_state=0)
    #     imp = IterativeSVD()
    #     # imp = SoftImpute()
    #     # imp = NuclearNormMinimization()
    #     drop_list = [generic_key, 'Gender']
    #     dfx = exposure_df.drop(drop_list, axis=1)
    #     features = dfx.columns
    #     X = dfx.values
    #     print(X.shape)
    #     y = exposure_df[drop_list].values
    #     print(y)
    #
    #     X = imp.fit_transform(X)
    #     exposure_df = DataFrame(X, columns=features)
    #     dropped_df = DataFrame(y, columns=drop_list)
    #     exposure_df[drop_list] = dropped_df
    #     # assert exposure_df.shape[0] == exposure_df.dropna(how='any').shape[0], "dfv.shape[0]: {}, dfv.dropna.shape[0]: {}".format(exposure_df.shape[0], exposure_df.dropna(how='any').shape[0])
    #
    # print(exposure_df.shape)
    ####################################################
    # B. load labels (from yet another file)
    # Richard: TODO: Adding confounding factors here to load
    primary_key = 'Study ID'
    age = "Today's age: (years)"
    label = 'Has a doctor ever diagnosed you with asthma?'
    # outcome = 'ACT score'

    # outcome = 'Have you ever needed to go to the emergency department for asthma?'
    # outcome = 'Have you ever needed to be hospitalized over night for asthma?'
    # outcome = 'Do you regularly use an asthma medication (one that has been prescribed by a physician for regular use)?'
    outcome = 'In the past 6 months, have you had regular (2/week) asthma symptoms?  '
    # outcome = 'In the past 6 mos, have you been on a daily controller asthma medication (e.g. Qvar, Flovent, Advair, Symbicort, Singulair)?'
    # outcome = 'At what age (in years) were you diagnosed with asthma? '
    # outcome = 'How many times in the past year have you gone to the emergency department for asthma? '
    # outcome = 'How many times in the past year have you been hospitalized over night?'
    # outcome = 'During the past week, how many days have you had asthma symptoms (e.g. shortness of breath, wheezing, chest tightness)?'

    gender = 'Gender'
    race = 'Race'

    # fever_sym = "Do you currently have hay fever symptoms (runny, stuffy nose accompanied by sneezing and itching when you did not have a cold or flu--sometimes called 'allergies')?"

    # only take these columns
    if outcome_limited_with_asthma:
        labelmap = {
            outcome: 'label',
            age: 'age',
            gender: 'gender',
            # race: 'race',
            # fever_sym: 'fever_symptom'
        }
        col_names = [primary_key, label, age, gender, outcome
                     # race
                     ]
        categorical_vars = [
            gender,
            # race
        ]
        if binary_outcome:
            categorical_vars.append(outcome)
    else:
        labelmap = {
            label: 'label',
            age: 'age',
            gender: 'gender',
            # race: 'race',
            # fever_sym: 'fever_symptom'
        }
        col_names = [primary_key, label, age, gender
                     # race
                     ]
        categorical_vars = [
            label,
            gender,
            # race
        ]

    # ... use the simpler new variable names?

    fpath_labels = os.path.join(dataDir, label_matrix)
    assert os.path.exists(fpath_labels), "(load_merge) Invalid path to the label matrix"
    asthma_df = pd.read_csv(fpath_labels, header=0, sep=sep)

    # Race need to be one-hot encoded
    one_hot_encoded_cols = ["Race"]

    one_hot_encoded_df = pd.get_dummies(asthma_df[one_hot_encoded_cols], prefix=one_hot_encoded_cols)

    zip_code_df = asthma_df[['What is your zip code?']]


    asthma_df = asthma_df[col_names]
    if outcome_limited_with_asthma:
        asthma_df = asthma_df[asthma_df[label] == 'Yes']
        asthma_df.drop(columns=[label], inplace=True)
    asthma_df.dropna(how='any', axis=0, inplace=True)
    # assert asthma_df.shape[1] == len(col_names)
    print(tabulate(asthma_df.head(5), headers='keys', tablefmt='psql'))

    # asthma_df.loc[asthma_df[outcome] < 5, outcome] = 0
    # asthma_df.loc[asthma_df[outcome] >= 5, outcome] = 1

    # rename
    # dfl = dfl.rename(columns={primary_key: generic_key, label: labelmap[label]})
    # print("(load_merge) dim(dfl): {} | cols:\n... {}\n".format(dfl.shape, dfl.columns.values))

    # transform
    msg = ''
    le = LabelEncoder()
    dflc = asthma_df[categorical_vars]
    msg += "... original dflc:\n"
    msg += tabulate(dflc.head(5), headers='keys', tablefmt='psql') + '\n'
    print(msg)

    # dflc2 = dflc.apply(le.fit_transform)  # log: '<' not supported between instances of 'str' and 'float'

    # transform column by column
    cols_encoded = {cvar: {} for cvar in categorical_vars}
    for cvar in categorical_vars:
        cvar_encoded = '{}_numeric'.format(cvar)
        asthma_df[cvar] = le.fit_transform(asthma_df[cvar].astype(str))

        # keep track of mapping

        cols_encoded[cvar] = {label: index for index, label in enumerate(le.classes_)}  # encoded value -> name

    # dfl[categorical_vars] = dflc2[categorical_vars]
    print(cols_encoded)
    # mapping
    if verify:
        msg += "> verifying the (value) encoding ...\n"
        for cvar in categorical_vars:
            mapping = cols_encoded[cvar]
            for label, index in mapping.items():
                print("... {} | {} -> {}".format(cvar, label, index))

                # rename
    colmap = {primary_key: generic_key}
    colmap.update(labelmap)
    msg += "> renaming columns via {}\n".format(colmap)
    asthma_df = asthma_df.rename(columns=colmap)
    msg += "... transformed dflc:\n"
    msg += tabulate(asthma_df.head(5), headers='keys', tablefmt='psql') + '\n'
    ##################################################
    # Richard: add to merge with avg income and race here
    zip_income_df = pd.read_csv(os.path.join(dataDir, avg_income_matrix))
    # zip_income_df = zip_income_df[['zip', 'Avg_inc']]
    zip_inc_mapping = dict(zip_income_df[['zip', 'Avg_Inc']].values)

    asthma_df['avg_income'] = zip_code_df['What is your zip code?'].map(zip_inc_mapping)

    asthma_df = pd.concat([asthma_df, one_hot_encoded_df], axis=1)
    asthma_df.dropna(inplace=True)
    ###################################################
    # print(exposure_df[generic_key])
    #
    # # exposure_df = pd.merge(exposure_df, asthma_df, on=generic_key, how='inner')
    # # exposure_df = pd.merge(exposure_df, pollutant_cat_df, on=generic_key, how='inner')
    # exposure_df = pd.merge(asthma_df, pollutant_cat_df, on=generic_key, how='inner')
    # exposure_df.dropna(how='any', inplace=True)
    # print(exposure_df.shape)
    #
    # # merge
    # msg += "> merging variable matrix and label matrix ...\n"
    #
    # # finally drop ID columns so that only explanatory and response variables remain
    # # exposure_df = exposure_df.drop([generic_key, 'birth_year'], axis=1)
    # exposure_df = exposure_df.drop(['birth_year'], axis=1)
    #
    # msg += "... final dim(vars_matrix) after including labels: {} | vars:\n... {}\n".format(exposure_df.shape,
    #                                                                                         exposure_df.columns.values)
    # # assert exposure_df.shape[0] == N0, "Some IDs do not have data? prior to join n={}, after join n={}".format(N0,
    # #                                                                                                            exposure_df.shape[
    # #                                                                                                                0])
    #
    # # example_cols = ['Gender', 'ECy1', 'SO4y1', 'NH4y1', 'Nity1', 'OCy1', 'label', ]
    # # msg += tabulate(exposure_df[example_cols].head(10), headers='keys', tablefmt='psql') + '\n'
    #
    # # msg += "... data after dropping NAs and/or imputation:\n"
    # # example_cols = ['Gender', 'ECy1', 'SO4y1', 'NH4y1', 'Nity1', 'OCy1', 'label', ]
    # # msg += tabulate(exposure_df[example_cols].head(10), headers='keys', tablefmt='psql') + '\n'
    #
    # # print(msg)

    if save:
        output_path = os.path.join(dataDir, output_matrix)
        asthma_df.to_csv(output_path, sep=sep, index=False, header=True)
        print('(load_merge) Saved output dataframe to: {}'.format(output_path))

    return asthma_df


def test(**kargs):
    time_windows_list = [-1, 0, range(-1, 1),
                         range(2), range(3), range(4), range(5),
                         range(-1, 2), range(-1, 3), range(-1, 4), range(-1, 5)]
    # output_name = 'regular_asthma_symptoms_past6months'
    # output_name = 'age_greaterthan5_diagnosed_asthma'
    # output_name = 'age_diagnosed_asthma'
    # output_name = 'emergency_dept'
    # output_name = 'emergency_dept_pastyr_count'
    # output_name = 'hospitalize_overnight'
    # output_name = 'hospitalize_overnight_pastyr_count'
    # output_name = 'regular_asthma_symptoms_daysCount_pastWeek'
    # output_name = 'regular_asthma_symptoms_past6months'
    # output_name = 'daily_controller_past6months'
    # output_name = 'regular_medication'
    # for time_window in time_windows_list:
    #     load_merge(vars_matrix='exposures-4yrs.csv',
    #                label_matrix='nasal_biomarker_asthma1019.csv',
    #                output_matrix='{}_7pollutants_no_impute{}.csv'.format(output_name, '{}'),
    #                # output_matrix='age_greaterthan5_diagnosed_asthma_7pollutants_no_impute{}.csv',
    #                tImputation = False, time_window=time_window,
    #                outcome_limited_with_asthma=True,
    #                binary_outcome=False)
    # output_name = 'asthma'
    # # output_name = 'asthma(act_score)'
    load_merge(
            # vars_matrix='exposures-4yrs.csv',
               label_matrix='nasal_biomarker_asthma1019.csv',
               output_matrix='{}_income.csv'.format(output_name, '{}'),
               # output_matrix='age_greaterthan5_diagnosed_asthma_7pollutants_no_impute{}.csv',
               tImputation=False, time_window=0,
               outcome_limited_with_asthma=False,
               binary_outcome=True)


    return

if __name__ == "__main__": 
    test()

