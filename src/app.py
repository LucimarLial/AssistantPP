from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sqlalchemy.orm import sessionmaker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils import get_table_download_link, hash_file_reference, FileReference

from utils import (detect_outliers, detects_unbalanced_classes, conditional_entropy, 
                   binning, scaling, standardization, onehot_encoder, ordinal_encoder,
                   over_sampling, under_sampling)

from utils import (markdown_outliers, markdown_missing_values, markdown_class_desbalance, 
                   markdown_class_desbalance_v2, markdown_class_desbalance_v3, markdown_binning,
                   markdown_scaling, markdown_standardization, markdown_onehot, markdown_ordinal)


from db import database, save_to_database_ORM, query_database_ORM_last_number_workflow, LogOperation



st.set_option('deprecation.showfileUploaderEncoding', False)


dict_db = {

    'name_operator': [
        'Data Outlier Treatment', 'Data Missing Imputation', 'Column Selection', 
        'Data Normalization', 'Data Standardization', 'Data Discretization', 
        'Data Coding', 'Data Type Convert', 'Data Unified', 
        'Oversampling', 'Undersampling', 'Houldout'
    ],

    'type_operator': ['Data Cleaning', 'Data Reduction', 'Data Sampling', 'Data Transformation', 'Data Partition'],

    'function_operator': [
        'DropOutlier', 'Imputation-1', 'Imputation0', 
        'ImputationAverage', 'ImputationMedian', 'ImputationModa', 
        'ImputationUnknown', 'LabelEncoder', 'DropQualitativeColumn', 
        'DropQuantitativeColumn', 'IncludeQualitativeColumn', 'IncludeQuantitativeColumn', 
        'KBinsDiscratizer', 'MinMaxScaler', 'StandardScaler', 
        'OneHotEncoder', 'OrdinalEncoder', 'SMOTE', 
        'RandomUnderSampler', 'UnifiedDatabase', 'TrainTestSplit'
    ]

}

conn_db = database(is_table_log=True)

last_number_workflow = query_database_ORM_last_number_workflow(conn=conn_db)


def main():
    # -------------------------------- Sidebar -------------------------------
    st.sidebar.markdown('## Load dataset')

    select_type = st.sidebar.selectbox('Choose the file extension', options=[
        'Select an option', 'csv', 'xlsx', 'database'
    ])

    
    sep_text_input = st.sidebar.text_input('Insert the selected file separator', value=',')
    encoding_text_input = st.sidebar.text_input('Enter the encoding of the selected file', value='None')
	
    file = st.sidebar.file_uploader('Uploader do arquivo', type=select_type)
    
    
    if select_type == 'database':
        user = st.sidebar.text_input('Inform the database user:')
        passwd = st.sidebar.text_input('Enter the password for the database:', type='password')
        db_ip = st.sidebar.text_input('Enter the IP address of the database:')
        db_name = st.sidebar.text_input('Enter the name of the database:')
        table_name = st.sidebar.text_input('Enter the name of the table:')


    # -------------------------- Main page content  ----------------
    # Uploading the file data
    @st.cache(allow_output_mutation=True)
    def read_file_data(file):
        if file is not None:
            if select_type == 'csv':
                df = pd.read_csv(file, sep=sep_text_input, encoding=encoding_text_input)
                return df
            elif select_type == 'xlsx':
                df = pd.read_excel(file)
                return df


    df = read_file_data(file)


    if not isinstance(df, pd.DataFrame):
        if select_type == 'database':
            if user and passwd and db_ip and db_name and table_name:
                conn = database(db_user=user, db_passwd=passwd, db_ip=db_ip, db_name=db_name, is_table_log=False)
                df = pd.read_sql_table(table_name, conn)


    if df is not None:
        
        # 1. Análise Exploratória de Dados
        st.title('   Data preprocessing assistant for classification problems')

        st.markdown('<br>'*2, unsafe_allow_html=True)

        database_name = st.text_input('Enter the name of the database:')
    
        exploration = pd.DataFrame({
            'column': df.columns, 'type': df.dtypes, 'NA #': df.isna().sum(), 'NA %': (df.isna().sum() / df.shape[0]) * 100
        })

        st.markdown('<br><br><br>', unsafe_allow_html=True)
        
        st.markdown('### 1 - Exploratory Data Analysis')
        st.markdown('#### 1.1 - Dataset information')
        if st.checkbox('Display raw data'):
            st.markdown('<br>', unsafe_allow_html=True)
            value = st.slider('Choose the number of lines:',
                              min_value=1, max_value=100, value=5)
            st.dataframe(df.head(value), width=900, height=600)
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
            st.markdown('** Dataset dimension**')
            st.markdown(df.shape)
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
            st.markdown('**Descriptive statistics of the quantitative columns**')
            st.dataframe(df.describe(), width=900, height=600)
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
            st.markdown(
                '**Dataset information: Column name, Type, Numbers of NaNs (null) and Percentage of NaNs**')
            st.dataframe(exploration, width=900, height=600)

  
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown('#### 1.2 - Distribution of quantitative and qualitative columns')
        #st.markdown('<br>', unsafe_allow_html=True)
        
        if st.checkbox('Plotar gráfico', key='21'):
            op6 = list(df.columns)
            op6.insert(0, 'Select an option')
            
            select_feature_quantitative = st.selectbox('Select a column', options=op6)
            
            if select_feature_quantitative not in 'Select an option':
                sns.countplot(y=select_feature_quantitative, data=df, orient='h')
                plt.title(str(select_feature_quantitative), fontsize=14)
                st.pyplot()
            else:
                pass
           
        
        
        # 2. Detect outliers 
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 2 - Data Cleaning')
        st.markdown('#### 2.1 - Detect and treat quantitative column outliers')
        
        op = list(df.select_dtypes(include=[np.number]).columns)
        op.insert(0, 'Select an option')
        
        select_boxplot = st.selectbox('Choose the column to plot a univariate boxplot:', options=op)
        
        if select_boxplot not in 'Select an option':
            if len(select_boxplot) > 0:
                colors = ['#B3F9C5']
                sns.boxplot(x=select_boxplot, data=df.select_dtypes(include=[np.number]), palette=colors)
                st.pyplot(dpi=100)
        else:
            st.markdown('**Boxplot chart - brief explanation:**')
            st.image('imgs/boxplot-information.png', width=700)
            
        st.markdown('<br>', unsafe_allow_html=True)
        
        if st.checkbox('Explanation of the method used'):
            st.markdown(markdown_outliers)
        
            st.markdown('<br>', unsafe_allow_html=True)
        
        is_remove_outliers_select = st.selectbox('Want to remove outliers?', options=(
            'Select an option', 'Yes', 'No'
        ))
        
        outliers_drop = detect_outliers(df, 2, list(exploration[exploration['type'] != 'object']['column'].index))
        
        
        if is_remove_outliers_select in 'Yes':

            df_copy = df.copy()

            df = df.drop(outliers_drop, axis = 0).reset_index(drop=True) # removing the outliers from the base
            st.dataframe(df_copy.loc[outliers_drop])
            st.write(df.shape)
            st.success('Outliers successfully removed!')

            name_column_list_outliers = df.columns.tolist()
            for col in name_column_list_outliers:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][0], name_operator=dict_db['name_operator'][0], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())


        # 3. Detect Missing values 
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('#### 2.2 - Detect and treat missing values')
        
        if st.checkbox('Explanation of missing values'):
        
            st.markdown(markdown_missing_values)
        
            st.markdown('<br>', unsafe_allow_html=True)
        
        percentual = st.slider(
            'Enter a missing value percentage limit:', min_value=0, max_value=100)
        
        op7 = list(df.columns)
        op7.insert(0, 'Select an option')
        columns_missing_to_remove = st.multiselect('Inform the columns you want to remove because they contain a large volume of missing values:', options=op7)
        
        num_columns_list = list(exploration[(exploration['NA %'] > percentual) & (
            exploration['type'] != 'object')]['column']) #quantitativa
        
        cat_columns_list = list(exploration[(exploration['NA %'] > percentual) & (
            exploration['type'] == 'object')]['column']) #qualitative
        
        
        if columns_missing_to_remove:
            df = df.drop(list(columns_missing_to_remove), axis=1).reset_index(drop=True)
            
            if num_columns_list:
                num_columns_list = [num_col for num_col in num_columns_list if num_col not in  columns_missing_to_remove]

                if len(num_columns_list) > 1:
                    for col in num_columns_list:
                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][9], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())
                else:
                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_columns_list, function_operator=dict_db['function_operator'][9], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

            
            if cat_columns_list:
                cat_columns_list = [cat_col for cat_col in cat_columns_list if cat_col not in columns_missing_to_remove]

                if len(cat_columns_list) > 1:
                    for col in cat_columns_list:
                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][8], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())
                else:
                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=cat_columns_list, function_operator=dict_db['function_operator'][8], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        
        st.markdown('<br>', unsafe_allow_html=True)

        # ---------------------------- Quantitative Columns --------------------
        
        st.markdown('#### Imputation of quantitative data')
        
        st.markdown(num_columns_list)

        imputer = st.selectbox('Choose an imputation option:', options=(
            'Select an option',
            'Input with -1',
            'Input with 0',
            'Input with average',
            'Input with median',
            'Input with moda',
            # 'Dropar'
        ))

        if imputer == 'Input with -1:
            df.fillna(-1, inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Values successfully filled!')

            name_column_list_imputer1 = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer1:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][1], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())


        elif imputer == 'Input with 0':
            df.fillna(0, inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Values successfully filled!')

            name_column_list_imputer0 = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer0:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][2], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        elif imputer == 'Input with average':
            df.fillna(
                df[num_columns_list].mean(), inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Values successfully filled!')

            name_column_list_imputer_avg = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer_avg:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][3], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        elif imputer == 'Input with median':
            df.fillna(
                df[num_columns_list].median(), inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Values successfully filled!')

            name_column_list_imputer_median = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer_median:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][4], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        elif imputer == 'Input with moda':
            df.fillna(
                df[num_columns_list].mode().iloc[0], inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Values successfully filled!')

            name_column_list_imputer_moda = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer_moda:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][5], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())


        # ------------------------- Qualitative Columns ---------------------
        st.markdown('<br>', unsafe_allow_html=True)
        
        st.markdown('#### Imputation of qualitative data')

        st.markdown(cat_columns_list)

        cat_imputer = st.selectbox('Choose an imputation option:', options=(
            'Select an option',
            'Input with unknown',
            # 'Dropar'
        ))

        if cat_imputer in 'Input with unknown':
            df.fillna('unknown', inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] == 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Values successfully filled!')

            name_column_list_impute_unk = df_no_missing_values.index.tolist()
            for col in name_column_list_impute_unk:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][6], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())


        #  Separate quantitative and qualitative variables
        
        num_features = df.select_dtypes(include=[np.number]).copy()
        cat_features = df.select_dtypes(exclude=[np.number]).copy()
        
        
        # 3. Check if the classes are unbalanced 
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 3 - Check imbalance between classes')
        
        if st.checkbox('Explanation of unbalance'):
        
            st.markdown(markdown_class_desbalance)
        
            st.markdown('<br>', unsafe_allow_html=True)
        
        op1 = list(df.columns)
        op1.insert(0, 'Select an option')
        
        select_target_desbalance = st.selectbox('Enter the target column:', options=op1)
        
        if st.checkbox('Plot graph'):
            if select_target_desbalance not in 'select an option':
                sns.countplot(x=select_target_desbalance, data=df) # plots a countplot chart to check the distribution of classes 
                plt.title('Target', fontsize=14)
                st.pyplot()
                    
                if detects_unbalanced_classes(df, select_target_desbalance) < 20.0:
                    st.markdown('<br>', unsafe_allow_html=True)
                    st.success('Classes with similar distribution, in fact balanced.')
                else:
                    st.markdown('<br>', unsafe_allow_html=True)
                    st.warning('Classes with the possibility of being unbalanced. The treatment in section 7 - Data Sampling Correction is recommended.')
                    
                st.markdown('<br>', unsafe_allow_html=True)
            
                    
                if df[select_target_desbalance].dtypes == 'object':
                    
                    st.warning('The target column is of the type qualitative - object. It is necessary to transform its type to quantitative.')
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    is_transformer_target_select = st.selectbox('Do you want to transform the target column to the quantitative type? (RECOMMENDED)', options=(
                        'Select an option', 'Yes', 'No'
                    ))
                    
                   
                    if is_transformer_target_select in 'Yes':
                        encoder = LabelEncoder()
                        df[select_target_desbalance] = encoder.fit_transform(df[select_target_desbalance])

                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=str(select_target_desbalance), function_operator=dict_db['function_operator'][7], name_operator=dict_db['name_operator'][7], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                        
                    
                    if df[select_target_desbalance].dtypes != 'object':
                        st.success('Successful transformation!')
                   

                    num_features[select_target_desbalance] = df[select_target_desbalance].copy()
                    
                    del cat_features[select_target_desbalance]
                    
                else:
                    pass
                    
                
            else:
                st.error('Enter a column!')
            
            
        
        # 4 - Correlation between quantitative columns 
        
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 4 - Data Reduction - Feature Selection')
        
        st.markdown('#### 4.1 - Correlation between columns')
        select_corr = st.selectbox(' 4.1.1 - Enter the correlation method between quantitative columns you want to analyze:', options=(
            'Select an option', 'pearson', 'kendall', 'spearman'
        ))

        if st.checkbox('Explanation of the correlation method'):
            st.markdown('''
				**Pearson's correlation**
				* Quantitative columns
				* Columns with normal distribution or sufficiently large sample
				* Preferable for linear type relationships

				**Correlação de Kerdell**
				* Ordinal scale columns 
				* Preferable when having small samples
    
				**Correlação de Spearman**
				* Quantitative or ordinal scale columns
				* Use when columns are not normal
				* Preferable when there is no linear relationship 
			''')
            
        st.markdown('<br>', unsafe_allow_html=True)

        if select_corr != 'select an option':
            if df.shape[1] <= 30:
                plt.rcParams['figure.figsize'] = (10, 8)
                sns.heatmap(num_features.corr(method=select_corr), annot=True,
                            linewidths=0.5, linecolor='black', cmap='Blues')
                st.pyplot(dpi=100)
            else:
                plt.rcParams['figure.figsize'] = (20, 10)
                sns.heatmap(num_features.corr(method=select_corr), annot=True,
                            linewidths=0.5, linecolor='black', cmap='Blues')
                st.pyplot(dpi=100)
                
        st.markdown('<br>', unsafe_allow_html=True)
        
      #  st.markdown('#### 4.1.1 - Correlation between quantitative columns')

        cat_features_delete = []
        
        if st.checkbox('Quantitative columns', key='1'):
            if st.checkbox('I want to use all columns', key='2'):

                if len(num_features.columns.tolist()) > 1:
                    for col in num_features.columns:
                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                else:
                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())


                st.success('All quantitative columns were selected!')
            else:
                
                num_fit_features_radio = st.radio('Do you want to include or exclude columns for preprocessing?', options=(
                    'Include', 'Exclude'
                ))
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                if num_fit_features_radio in 'Include':
                    num_fit_features_add = st.multiselect(
                        'Select columns to include', options=list(df.select_dtypes(include=[np.number]).columns))
                    num_features = num_features[num_fit_features_add]

                    
                    if len(num_fit_features_add) > 1:
                        for col in num_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                    else:
                        if num_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_fit_features_add[0], function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Selected columns -> {list(num_features.columns)}')
                
                if num_fit_features_radio in 'Exclude':
                    num_fit_features_delete = st.multiselect(
                        'Select columns to exclude', options=list(df.select_dtypes(include=[np.number]).columns)
                    )
                    num_features = num_features.drop(num_fit_features_delete, axis=1)


                    if len(num_fit_features_delete) > 1:
                        for col in num_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][9], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                    else:
                        if num_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_fit_features_delete[0], function_operator=dict_db['function_operator'][9], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Available columns -> {list(num_features.columns)}')
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        st.markdown('#### 4.1.2 - The correlation between qualitative columns is based on the calculation of entropy')
        st.markdown('<br>', unsafe_allow_html=True)
                           
        if st.checkbox('Qualitative columns', key='3'):
            if st.checkbox('I want to use all columns', key='4'):

                if len(cat_features.columns.tolist()) > 1:
                    for col in cat_features.columns:
                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                else:
                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=cat_features.columns.tolist()[0], function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())


                st.success('All qualitative columns have been selected!')
            else: 
                op2 = list(df.columns)
                op2.insert(0, 'select an option')
                
                select_cat_corr = st.selectbox(
                    'Enter the target column to calculate the correlation with the qualitative columns', options=op2)

                cat_corr = {}

                if select_cat_corr != 'select an option':
                    for col in cat_features.columns:
                        cat_corr[col] = conditional_entropy(
                            cat_features[col], df[select_cat_corr])

                series_cat_corr = pd.Series(cat_corr, name='correlation')
                st.dataframe(series_cat_corr)

                
                cat_fit_features_add_radio = st.radio('Do you want to include or exclude columns for preprocessing?', options=(
                    'Include', 'Exclude'
                ), key='1')
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                if cat_fit_features_add_radio in 'Include':
                    cat_fit_features_add = st.multiselect(
                    'Select columns to include', options=list(cat_features.columns))
                    
                    cat_features = cat_features[cat_fit_features_add]

                    if len(cat_fit_features_add) > 1:
                        for col in cat_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    else:
                        if cat_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=cat_fit_features_add[0], function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Selected columns -> {list(cat_features.columns)}')
                
                elif cat_fit_features_add_radio in 'Exclude':
                    cat_fit_features_delete = st.multiselect(
                        'Select columns to exclude', options=list(cat_features.columns)
                    )
                    cat_features_delete.append(cat_fit_features_delete)
                    
                    cat_features = cat_features.drop(cat_fit_features_delete, axis=1)

                    if len(cat_fit_features_delete) > 1:
                        for col in cat_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][8], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    else:
                        if cat_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=cat_fit_features_delete[0], function_operator=dict_db['function_operator'][8], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Available columns -> {list(cat_features.columns)}')
                            
        
        
        # 5 - Feature engineering       
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 5 - Data Transformation -  Feature Engineering')
        
        op3 = list(num_features.columns)
        op3.insert(0, 'Select an option ')
        
        select_target = st.selectbox('Enter the target column:', options=list(op3))
        
        st.markdown('<br>', unsafe_allow_html=True)
        
              
        if select_target not in 'Select an option':
            
            
            if st.checkbox('Quantitative columns', key='5'):
                
                is_applied_binning = False
                
                if st.checkbox('Contínuas', key='6'):
                    if st.checkbox('Explanation of Discretization'):        
                        st.markdown(markdown_binning)
            
                    n_bins_slider = st.slider('n_bins', min_value=2, max_value=20, value=5)
                    encode_select = st.selectbox('encode', options=('onehot-dense', 'ordinal'))
                    strategy_select = st.selectbox('strategy', options=('quantile', 'uniform', 'kmeans'))
                    
                    
                    select_col_binning = st.multiselect('Inform the columns you want to apply Discretization:', options=list(num_features.drop(select_target, axis=1).columns))
                    list_col_binning = list(select_col_binning)
                    st.markdown(list_col_binning)
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    num_features[select_col_binning] = binning(num_features[select_col_binning], n_bins=n_bins_slider, encode=encode_select, strategy=strategy_select)
                    
                    is_applied_binning = True


                    if len(select_col_binning) > 1:
                        for col in select_col_binning:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][12], name_operator=dict_db['name_operator'][5], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                    else:
                        if select_col_binning:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=select_col_binning[0], function_operator=dict_db['function_operator'][12], name_operator=dict_db['name_operator'][5], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                    
                    if list_col_binning:
                        st.success('Successful transformation!')
                
                if st.checkbox('Discreet and Continuous'):
                    if st.checkbox('Explanation of Normalization and Standardization'):   
                        st.markdown(markdown_scaling)
                        st.markdown('<br>', unsafe_allow_html=True)
                        st.markdown(markdown_standardization)
                        
                    select_method_var_quantitative = st.selectbox('Choose the method:', options=('Select an option ', 'Normalization', 'Standardization'))
                    
                    if select_method_var_quantitative in 'Normalization':
                        if is_applied_binning:
                            num_features = pd.concat([
                                scaling(num_features.drop(select_col_binning, axis=1), select_target),
                                num_features[select_col_binning]
                            ], axis=1).reset_index(drop=True)

                            if len(num_features.columns.tolist()) > 1:
                                for col in num_features.columns:
                                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][13], name_operator=dict_db['name_operator'][3], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            else:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][13], name_operator=dict_db['name_operator'][3], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            
                            st.success('Successful transformation!')
                        else:
                            num_features = scaling(num_features, select_target)

                            if len(num_features.columns.tolist()) > 1:
                                for col in num_features.columns:
                                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][13], name_operator=dict_db['name_operator'][3], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            else:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][13], name_operator=dict_db['name_operator'][3], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            st.success('Successful transformation!')
                        
                    if select_method_var_quantitative in 'Standardization':
                        if is_applied_binning:
                            num_features = pd.concat([
                                standardization(num_features.drop(select_col_binning, axis=1), select_target),
                                num_features[select_col_binning]
                            ], axis=1).reset_index(drop=True)


                            if len(num_features.columns.tolist()) > 1:
                                for col in num_features.columns:
                                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][14], name_operator=dict_db['name_operator'][4], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            else:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][14], name_operator=dict_db['name_operator'][4], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            
                            st.success('Successful transformation!')
                        else:
                            num_features = standardization(num_features, select_target)

                            if len(num_features.columns.tolist()) > 1:
                                for col in num_features.columns:
                                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][14], name_operator=dict_db['name_operator'][4], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            else:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][14], name_operator=dict_db['name_operator'][4], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            st.success('Successful transformation!')
            
            st.markdown('<br>', unsafe_allow_html=True)
             
                       
            if st.checkbox('Qualitative columns', key='7'):
                
                is_onehot_transform = False
                is_ordinal_transform = False
                
                no_preprocessed = []
                
                if st.checkbox('Nominal (OneHot Encoder)', key='8'):
                    if st.checkbox('Coding explanation - OneHot Encoder'):  
                        st.markdown(markdown_onehot)
                        st.markdown('<br>', unsafe_allow_html=True)
                    
                    select_cat_features_nominal = st.multiselect('Inform the columns you want to apply the encoding - OneHot Encoder:', options=list(cat_features.columns))
                    list_cat_features_nominal = list(select_cat_features_nominal)
                    st.markdown(list_cat_features_nominal)
                    
                    onehot_transform = onehot_encoder(cat_features[select_cat_features_nominal])
                    
                    is_onehot_transform = True

                    if len(select_cat_features_nominal) > 1:
                        for col in select_cat_features_nominal:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][15], name_operator=dict_db['name_operator'][6], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                    else:
                        if select_cat_features_nominal:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=select_cat_features_nominal[0], function_operator=dict_db['function_operator'][15], name_operator=dict_db['name_operator'][6], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                    
                    if list_cat_features_nominal:
                        no_preprocessed.extend(list_cat_features_nominal)
                        st.success('Transformação realizada com sucesso!')
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                if st.checkbox('Ordinal (Ordinal Encoder)', key='9'):
                    if st.checkbox('Explicação da codificação - Ordinal Encoder'):    
                        st.markdown(markdown_ordinal)
                    
                        st.markdown('<br>', unsafe_allow_html=True)
                    
                    select_cat_features_ordinal = st.multiselect('Informe as colunas que deseja aplicar a codificação - Ordinal Encoder:', options=list(cat_features.columns))
                    list_cat_features_ordinal = list(select_cat_features_ordinal)
                    st.markdown(list_cat_features_ordinal)
                    
                    ordinal_transform = ordinal_encoder(cat_features[select_cat_features_ordinal])
                    
                    is_ordinal_transform = True

                    if len(select_cat_features_ordinal) > 1:
                        for col in select_cat_features_ordinal:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][16], name_operator=dict_db['name_operator'][6], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())
                    else:
                        if select_cat_features_ordinal:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=select_cat_features_ordinal[0], function_operator=dict_db['function_operator'][16], name_operator=dict_db['name_operator'][6], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                    
                    if list_cat_features_ordinal:
                        no_preprocessed.extend(list_cat_features_ordinal)
                        st.success('Successful transformation!')
                
                if is_onehot_transform:
                    cat_features = onehot_transform
                if is_ordinal_transform:
                    cat_features = ordinal_transform
                if is_onehot_transform and is_ordinal_transform:
                    cat_features = pd.concat([onehot_transform, ordinal_transform], axis=1).reset_index(drop=True)
                    

                cat_cols = list(df.select_dtypes(include=np.object).columns)

                no_preprocessed = [no for no in no_preprocessed]
 
                col_no_preprocessed = [col for col in cat_cols if col not in no_preprocessed]
                
                if col_no_preprocessed:
                    cat_features = pd.concat([cat_features, df[col_no_preprocessed]], axis=1).reset_index(drop=True)
                
                if cat_features_delete:
                    cat_features = cat_features.drop(cat_features_delete[0], axis=1)
                                
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
     
            
        # 8 - Generate preprocessed files (Trainee and test) / Single Base 
        st.markdown('### 6 - Data Partition - Training and Testing or Single Base')
        
        
        is_select_partition = st.selectbox('Do you want to partition the dataset in Training and Testing?', options=('Select an option', 'No', 'Yes'))
        st.markdown('<br>', unsafe_allow_html=True)
        
        
        if is_select_partition in 'Select an option':
            pass
        else:
            if is_select_partition in 'No':
                op4 = list(num_features.columns)
                op4.insert(0, 'Select an option')
                
                select_target_partition = st.selectbox('Inform the target column to carry out the partitioning of the dataset:', options=list(op4))
                            
                
                if select_target_partition is not 'Select an option':
                    if select_target_partition not in cat_features:
                        X = pd.concat([num_features.drop(select_target_partition, axis=1).reset_index(drop=True), cat_features], axis=1).reset_index(drop=True)
                        
                    y = num_features[select_target_partition].copy()
                    
                    unified_base = pd.concat([X, y], axis=1).reset_index(drop=True)
                    
                    
                    if not unified_base.empty:
                        st.success('Base disponível para download')
                    
                
                    is_completed_select = st.sidebar.selectbox('Completed all preprocessing operations ?', options=('No', 'Yes'))
                    
                    
                    if is_completed_select in 'Yes':
                        bs4_unified = get_table_download_link(unified_base)
                        
                        st.sidebar.markdown(f'''
                            <style>
                                .button {{
                                    display: block;
                                    padding: 8px 16px;
                                    font-size: 16px;
                                    cursor: pointer;
                                    text-align: center;
                                    text-decoration: none;
                                    outline: none;
                                    color: #fff;
                                    background-color:rgba(246, 51, 102, .6);
                                    border: none;
                                    border-radius: 15px;
                                    box-shadow: 0 9px #999;
                                    margin:auto;
                                }}
                                .button:hover {{
                                    background-color:rgba(246, 51, 102);
                                }}
                                .button:active {{
                                    background-color:rgba(246, 51, 102);
                                    box-shadow: 0 5px #666;
                                    transform: translateY(4px);
                                }}
                            </style>
                            
                            <a style="text-decoration:none;" href="data:file/csv;base64,{bs4_unified}" download="Base unificada.csv"><button class="button">Base Única</button></a>
                        ''', unsafe_allow_html=True)


                        if len(unified_base.columns.tolist()) > 1:
                            for col in unified_base.columns:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][19], name_operator=dict_db['name_operator'][8], type_operator=dict_db['type_operator'][4], timestamp=datetime.now())
                        else:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=unified_base.columns.tolist()[0], function_operator=dict_db['function_operator'][19], name_operator=dict_db['name_operator'][8], type_operator=dict_db['type_operator'][4], timestamp=datetime.now())



                    else:
                        st.sidebar.warning('Completed preprocessing?')
            
            
            
            if is_select_partition in 'Yes':
            
                op5 = list(num_features.columns)
                op5.insert(0, 'Select an option')
                
                select_target_partition = st.selectbox('Enter the target column to generate the pre-processed dataset:', options=list(op5))
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                
                
                
                if select_target_partition != 'Select an option':
                    if select_target_partition not in cat_features:
                        X = pd.concat([num_features.drop(select_target_partition, axis=1).reset_index(drop=True), cat_features], axis=1).reset_index(drop=True)
                        
                    y = num_features[select_target_partition].copy()
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    
                    select_test_size = st.slider('Inform the proportion of the size of the test base:', min_value=1, max_value=99, value=25)
                    
                    st.write(X.shape)
                    st.write(y.shape)
                        
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(select_test_size) / 100.0)
                    
                    
                    st.write(float(select_test_size) / 100.0)
                    
                    st.markdown('**Training**')
                    st.write(X_train.shape)
                    st.write(y_train.shape)
                    
                    st.markdown('**Testing**')
                    st.write(X_test.shape)
                    st.write(y_test.shape)

                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    
                    # 7 - Correction of Data Sampling 
                    st.markdown('### 7 - Correction of Data Sampling')
                                      
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    if st.checkbox('Explanation of data sampling correction'):  
                        st.markdown(markdown_class_desbalance_v2)
                        st.markdown('<br>', unsafe_allow_html=True)
                    
                    if st.checkbox('Explanation of the method to be used'):
                        st.markdown(markdown_class_desbalance_v3)
                    
                    method_balance_select = st.selectbox('Choose the most appropriate method for your problem:', options=(
                        'Select an option', 'Oversampling', 'Undersampling'
                    ))
                    
                    if method_balance_select in 'Oversampling':
                        try:
                            X_train, y_train = over_sampling(X_train, y_train)

                            sampling_cols = X_train.columns.tolist() + [y.name]
                            for col in sampling_cols:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][17], name_operator=dict_db['name_operator'][9], type_operator=dict_db['type_operator'][2], timestamp=datetime.now())

                            st.success('Oversampling successfully applied!')
                        except Exception as e:
                            st.markdown(e)              
                        
                    if method_balance_select in 'Undersampling':
                        X_train, y_train = under_sampling(X_train, y_train)

                        under_sampling_cols = X_train.columns.tolist() + [y.name]
                        for col in under_sampling_cols:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][18], name_operator=dict_db['name_operator'][10], type_operator=dict_db['type_operator'][2], timestamp=datetime.now())
                        
                        st.success('Undersampling successfully applied!')
                
                
                    train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
                    test = X_test
                
                    is_completed_select = st.sidebar.selectbox('Completed all preprocessing operations?', options=('No', 'Yes'))
                    
                    
                    if is_completed_select in 'Yes':
                        bs4_train = get_table_download_link(train)
                        
                        st.sidebar.markdown(f'''
                            <style>
                                .button {{
                                    display: block;
                                    padding: 8px 16px;
                                    font-size: 16px;
                                    cursor: pointer;
                                    text-align: center;
                                    text-decoration: none;
                                    outline: none;
                                    color: #fff;
                                    background-color:rgba(246, 51, 102, .6);
                                    border: none;
                                    border-radius: 15px;
                                    box-shadow: 0 9px #999;
                                    margin:auto;
                                }}
                                .button:hover {{
                                    background-color:rgba(246, 51, 102);
                                }}
                                .button:active {{
                                    background-color:rgba(246, 51, 102);
                                    box-shadow: 0 5px #666;
                                    transform: translateY(4px);
                                }}
                            </style>
                            
                            <a style="text-decoration:none;" href="data:file/csv;base64,{bs4_train}" download="treino.csv"><button class="button">Treino</button></a>
                        ''', unsafe_allow_html=True)

                    
                    
                        bs4_test = get_table_download_link(test)
                        
                        st.sidebar.markdown(f'''
                            <style>
                                .button {{
                                    display: block;
                                    padding: 8px 16px;
                                    font-size: 16px;
                                    cursor: pointer;
                                    text-align: center;
                                    text-decoration: none;
                                    outline: none;
                                    color: #fff;
                                    background-color:rgba(246, 51, 102, .6);
                                    border: none;
                                    border-radius: 15px;
                                    box-shadow: 0 9px #999;
                                    margin:auto;
                                }}
                                .button:hover {{
                                    background-color:rgba(246, 51, 102);
                                }}
                                .button:active {{
                                    background-color:rgba(246, 51, 102);
                                    box-shadow: 0 5px #666;
                                    transform: translateY(4px);
                                }}
                            </style>
                            
                            <a style="text-decoration:none;" href="data:file/csv;base64,{bs4_test}" download="teste.csv"><button class="button">Teste</button></a>
                        ''', unsafe_allow_html=True)


                        partition_cols = train.columns.tolist() + test.columns.tolist()
                        for col in partition_cols:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][20], name_operator=dict_db['name_operator'][11], type_operator=dict_db['type_operator'][4], timestamp=datetime.now())

                    else:
                        st.sidebar.warning('Have you completed pre-processing?')
                     
        
    else:

        st.sidebar.markdown('## Workflow query ')
        select_query_workflow = st.sidebar.selectbox('', options=('Selecione uma opção', 'Fazer consulta'))

        if select_query_workflow != 'Fazer consulta':
            st.markdown('<h1 align="center"> Data PreProcessing Assistant for Classification Problems </h1>', unsafe_allow_html=True)
            st.image('imgs/capa.png')

        if select_query_workflow == 'Fazer consulta':
            query = st.text_area('Query input')

            if query:
                try:
                    value_tal = st.slider('', min_value=1, max_value=1000, value=5)
                    df_query = pd.read_sql(query, conn_db)

                    st.table(df_query.head(value_tal))
                except Exception as e:
                    st.error('Invalid Query!')


        

