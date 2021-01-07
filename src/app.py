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
        'RandomUnderSampler', 'Unified database', 'TrainTestSplit'
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

    
    sep_text_input = st.sidebar.text_input('Enter the separator of the selected file', value=',')
    encoding_text_input = st.sidebar.text_input('Enter the encoding of the selected file', value='None')

    file = st.sidebar.file_uploader('File uploader', type=select_type)
    
    
    if select_type == 'banco de dados':
        user = st.sidebar.text_input('Informe o usuário do banco de dados:')
        passwd = st.sidebar.text_input('Informe a senha do banco de dados:', type='password')
        db_ip = st.sidebar.text_input('Informe o IP de endereço do banco de dados')
        db_name = st.sidebar.text_input('Informe o nome do banco de dados')
        table_name = st.sidebar.text_input('Informe o nome da tabela')


    # -------------------------- Conteúdo da página principal ----------------
    # Carregando os dados de arquivo
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
        if select_type == 'banco de dados':
            if user and passwd and db_ip and db_name and table_name:
                conn = database(db_user=user, db_passwd=passwd, db_ip=db_ip, db_name=db_name, is_table_log=False)
                df = pd.read_sql_table(table_name, conn)


    if df is not None:
        
        # 1. Análise Exploratória de Dados
        st.title('   Data Pre-Processing Assistant for Classification Problems')

        st.markdown('<br>'*2, unsafe_allow_html=True)

        database_name = st.text_input('Informe o nome da base de dados:')
    
        exploration = pd.DataFrame({
            'column': df.columns, 'type': df.dtypes, 'NA #': df.isna().sum(), 'NA %': (df.isna().sum() / df.shape[0]) * 100
        })

        st.markdown('<br><br><br>', unsafe_allow_html=True)
        
        st.markdown('### 1 - Análise Exploratória de Dados')
        st.markdown('#### 1.1 - Informações do conjunto de dados')
        if st.checkbox('Exibir dados brutos'):
            st.markdown('<br>', unsafe_allow_html=True)
            value = st.slider('Escolha o número de linhas:',
                              min_value=1, max_value=100, value=5)
            st.dataframe(df.head(value), width=900, height=600)
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
            st.markdown('** Dimensão do conjunto de dados**')
            st.markdown(df.shape)
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
            st.markdown('**Estatística descritiva das colunas quantitativas**')
            st.dataframe(df.describe(), width=900, height=600)
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
            st.markdown(
                '**Informações do conjunto de dados: Nome da coluna, Tipo, Números de NaNs (nulos) e Porcentagem de NaNs**')
            st.dataframe(exploration, width=900, height=600)

  
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown('#### 1.2 - Distribuição das colunas quantitativas e qualitativas')
        #st.markdown('<br>', unsafe_allow_html=True)
        
        if st.checkbox('Plotar gráfico', key='21'):
            op6 = list(df.columns)
            op6.insert(0, 'Selecione uma opção')
            
            select_feature_quantitative = st.selectbox('Selecione uma coluna', options=op6)
            
            if select_feature_quantitative not in 'Selecione uma opção':
                sns.countplot(y=select_feature_quantitative, data=df, orient='h')
                plt.title(str(select_feature_quantitative), fontsize=14)
                st.pyplot()
            else:
                pass
           
        
        
        # 2. Detectar outliers
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 2 - Limpeza de Dados')
        st.markdown('#### 2.1 - Detectar e tratar outliers das colunas quantitativas')
        
        op = list(df.select_dtypes(include=[np.number]).columns)
        op.insert(0, 'Selecione uma opção')
        
        select_boxplot = st.selectbox('Escolha a coluna para plotar um boxplot univariado:', options=op)
        
        if select_boxplot not in 'Selecione uma opção':
            if len(select_boxplot) > 0:
                colors = ['#B3F9C5']
                sns.boxplot(x=select_boxplot, data=df.select_dtypes(include=[np.number]), palette=colors)
                st.pyplot(dpi=100)
        else:
            st.markdown('**Gráfico boxplot - breve explicação:**')
            st.image('imgs/boxplot-information.png', width=700)
            
        st.markdown('<br>', unsafe_allow_html=True)
        
        if st.checkbox('Explicação do método utilizado'):
            st.markdown(markdown_outliers)
        
            st.markdown('<br>', unsafe_allow_html=True)
        
        is_remove_outliers_select = st.selectbox('Deseja remover outliers?', options=(
            'Selecione uma opção', 'Sim', 'Não'
        ))
        
        outliers_drop = detect_outliers(df, 2, list(exploration[exploration['type'] != 'object']['column'].index))
        
        
        if is_remove_outliers_select in 'Sim':

            df_copy = df.copy()

            df = df.drop(outliers_drop, axis = 0).reset_index(drop=True) # removendo os outliers da base
            st.dataframe(df_copy.loc[outliers_drop])
            st.write(df.shape)
            st.success('Outliers removido com sucesso!')

            name_column_list_outliers = df.columns.tolist()
            for col in name_column_list_outliers:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][0], name_operator=dict_db['name_operator'][0], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())


        # 3. Detectar Missing values
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('#### 2.2 - Detectar e tratar valores ausentes (missing values)')
        
        if st.checkbox('Explicação de missing values'):
        
            st.markdown(markdown_missing_values)
        
            st.markdown('<br>', unsafe_allow_html=True)
        
        percentual = st.slider(
            'Informe um limite de percentual de valor faltante:', min_value=0, max_value=100)
        
        op7 = list(df.columns)
        op7.insert(0, 'Selecione uma opção')
        columns_missing_to_remove = st.multiselect('Informe as colunas que deseja remover, por conter grande volume de valores ausentes:', options=op7)
        
        num_columns_list = list(exploration[(exploration['NA %'] > percentual) & (
            exploration['type'] != 'object')]['column'])
        
        cat_columns_list = list(exploration[(exploration['NA %'] > percentual) & (
            exploration['type'] == 'object')]['column'])
        
        
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

        # ---------------------------- Variáveis Numéricas --------------------
        
        st.markdown('#### Imputação dos dados quantitativos')
        
        st.markdown(num_columns_list)

        imputer = st.selectbox('Escolha uma opção de imputação:', options=(
            'Selecione uma opção',
            'Imputar com -1',
            'Imputar com 0',
            'Imputar pela média',
            'Imputar pela mediana',
            'Imputar pela moda',
            # 'Dropar'
        ))

        if imputer == 'Imputar com -1':
            df.fillna(-1, inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Valores preenchidos com sucesso!')

            name_column_list_imputer1 = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer1:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][1], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())


        elif imputer == 'Imputar com 0':
            df.fillna(0, inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Valores preenchidos com sucesso!')

            name_column_list_imputer0 = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer0:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][2], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        elif imputer == 'Imputar pela média':
            df.fillna(
                df[num_columns_list].mean(), inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Valores preenchidos com sucesso!')

            name_column_list_imputer_avg = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer_avg:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][3], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        elif imputer == 'Imputar pela mediana':
            df.fillna(
                df[num_columns_list].median(), inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Valores preenchidos com sucesso!')

            name_column_list_imputer_median = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer_median:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][4], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        elif imputer == 'Imputar pela moda':
            df.fillna(
                df[num_columns_list].mode().iloc[0], inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Valores preenchidos com sucesso!')

            name_column_list_imputer_moda = df_no_missing_values.index.tolist()
            for col in name_column_list_imputer_moda:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][5], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        # elif imputer == 'Dropar':
        #     df.dropna(axis=0, inplace=True)
        #     na_dict = { 'NA %' : df[exploration[(exploration['NA %'] > 0) & (exploration['type'] != 'object')]['column']].isna().sum() }
        #     df_no_missing_values = pd.DataFrame(na_dict)
        #     st.write(df.shape)
        #     st.dataframe(df_no_missing_values.T)
        #     st.markdown('**Valores preenchidos/removidos com sucesso!**')

        # ------------------------- Variáveis Categóricas ---------------------
        st.markdown('<br>', unsafe_allow_html=True)
        
        st.markdown('#### Imputação dos dados qualitativos')

        st.markdown(cat_columns_list)

        cat_imputer = st.selectbox('Escolha uma opção de imputação:', options=(
            'Selecione uma opção',
            'Imputar com unknown',
            # 'Dropar'
        ))

        if cat_imputer in 'Imputar com unknown':
            df.fillna('unknown', inplace=True)
            na_dict = { 'NA %' : df[exploration[(exploration['NA %'].drop(columns_missing_to_remove) > 0) & (exploration['type'] == 'object')]['column']].isna().sum() }
            df_no_missing_values = pd.DataFrame(na_dict)
            st.dataframe(df_no_missing_values.T)
            st.success('Valores preenchidos com sucesso!')

            name_column_list_impute_unk = df_no_missing_values.index.tolist()
            for col in name_column_list_impute_unk:
                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][6], name_operator=dict_db['name_operator'][1], type_operator=dict_db['type_operator'][0], timestamp=datetime.now())

        # elif cat_imputer in 'Dropar':
        #     df.dropna(axis=0, inplace=True)
        #     na_dict = { 'NA %' : df[exploration[(exploration['NA %'] > 0) & (exploration['type'] == 'object')]['column']].isna().sum() }
        #     df_no_missing_values = pd.DataFrame(na_dict)
        #     st.write(df.shape)
        #     st.dataframe(df_no_missing_values.T)
        #     st.markdown('**Valores preenchidos/removidos com sucesso!**')



        #  Separar variáveis quantitativas e qualitativas
        
        num_features = df.select_dtypes(include=[np.number]).copy()
        cat_features = df.select_dtypes(exclude=[np.number]).copy()
        
        
        # 3. Verificar se as classes estão desbalanceadas
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 3 - Verificar Desbalanceamento entre Classes')
        
        if st.checkbox('Explicação de desbalanceamento'):
        
            st.markdown(markdown_class_desbalance)
        
            st.markdown('<br>', unsafe_allow_html=True)
        
        op1 = list(df.columns)
        op1.insert(0, 'Selecione uma opção')
        
        select_target_desbalance = st.selectbox('Informe a coluna alvo:', options=op1)
        
        if st.checkbox('Plotar gráfico'):
            if select_target_desbalance not in 'Selecione uma opção':
                sns.countplot(x=select_target_desbalance, data=df) # plota um gráfico countplot para verificar a distribuição das classes 
                plt.title('Target', fontsize=14)
                st.pyplot()
                    
                if detects_unbalanced_classes(df, select_target_desbalance) < 20.0:
                    st.markdown('<br>', unsafe_allow_html=True)
                    st.success('Classes com comportamentos próximos, de fato balanceada.')
                else:
                    st.markdown('<br>', unsafe_allow_html=True)
                    st.warning('Classes com possibilidade de estarem desbalanceadas. Recomenda-se o tratamento na seção 7 - Correção  da Amostragem de Dados.')
                    
                st.markdown('<br>', unsafe_allow_html=True)
            
                    
                if df[select_target_desbalance].dtypes == 'object':
                    
                    st.warning('A coluna target é do tipo qualitativo - object. Necessário transformar seu tipo para quantitativo.')
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    is_transformer_target_select = st.selectbox('Deseja transformar a coluna target para o tipo quantitativo? (RECOMENDADO)', options=(
                        'Selecione uma opção', 'Sim', 'Não'
                    ))
                    
                   
                    if is_transformer_target_select in 'Sim':
                        encoder = LabelEncoder()
                        df[select_target_desbalance] = encoder.fit_transform(df[select_target_desbalance])

                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=str(select_target_desbalance), function_operator=dict_db['function_operator'][7], name_operator=dict_db['name_operator'][7], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                        
                    
                    if df[select_target_desbalance].dtypes != 'object':
                        st.success('Transformação realizada com sucesso!')
                   

                    num_features[select_target_desbalance] = df[select_target_desbalance].copy()
                    
                    del cat_features[select_target_desbalance]
                    
                else:
                    pass
                    
                
            else:
                st.error('Informe uma coluna!')
            
            
        
        # 4 - Correlação entre as variáveis quantitativas
        
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 4 - Redução de Dados - Feature Selection')
        
        st.markdown('#### 4.1 - Correlação entre as colunas')
        select_corr = st.selectbox(' 4.1.1 - Informe o método de correlação entre colunas quantitativas que deseja analisar:', options=(
            'Selecione uma opção', 'pearson', 'kendall', 'spearman'
        ))

        if st.checkbox('Explicação do método de correlação'):
            st.markdown('''
				**Correlação de Pearson**
				* Colunas quantitativas
				* Colunas com distribuição normal ou amostra suficientimente grande
				* Preferível para relações do tipo linear

				**Correlação de Kerdell**
				* Colunas em escala ordinal
				* Preferível quando se têm amostras pequenas
    
				**Correlação de Spearman**
				* Colunas quantitativas ou em escala ordinal
				* Utilizar quando não se tem a normalidade das colunas
				* Preferível quando não se tem uma relação linear
			''')
            
        st.markdown('<br>', unsafe_allow_html=True)

        if select_corr != 'Selecione uma opção':
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
        
      #  st.markdown('#### 4.1.1 - Correlação entre as colunas quantitativas')

        cat_features_delete = []
        
        if st.checkbox('Colunas quantitativas', key='1'):
            if st.checkbox('Quero usar todas as colunas', key='2'):

                if len(num_features.columns.tolist()) > 1:
                    for col in num_features.columns:
                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                else:
                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())


                st.success('Todas as colunas quantitativas foram selecionadas!')
            else:
                
                num_fit_features_radio = st.radio('Deseja incluir ou excluir colunas para o pré-processamento?', options=(
                    'Incluir', 'Excluir'
                ))
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                if num_fit_features_radio in 'Incluir':
                    num_fit_features_add = st.multiselect(
                        'Selecione as colunas para inclusão', options=list(df.select_dtypes(include=[np.number]).columns))
                    num_features = num_features[num_fit_features_add]

                    
                    if len(num_fit_features_add) > 1:
                        for col in num_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                    else:
                        if num_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_fit_features_add[0], function_operator=dict_db['function_operator'][11], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Colunas selecionadas -> {list(num_features.columns)}')
                
                if num_fit_features_radio in 'Excluir':
                    num_fit_features_delete = st.multiselect(
                        'Selecione as colunas para exclusão', options=list(df.select_dtypes(include=[np.number]).columns)
                    )
                    num_features = num_features.drop(num_fit_features_delete, axis=1)


                    if len(num_fit_features_delete) > 1:
                        for col in num_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][9], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                    else:
                        if num_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_fit_features_delete[0], function_operator=dict_db['function_operator'][9], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Colunas disponíveis -> {list(num_features.columns)}')
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        st.markdown('#### 4.1.2 - A correlação entre as colunas qualitativas é baseada no cálculo da entropia')
        st.markdown('<br>', unsafe_allow_html=True)
                           
        if st.checkbox('Colunas qualitativas', key='3'):
            if st.checkbox('Quero usar todas as colunas', key='4'):

                if len(cat_features.columns.tolist()) > 1:
                    for col in cat_features.columns:
                        save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())
                else:
                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=cat_features.columns.tolist()[0], function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())


                st.success('Todas as colunas qualitativas foram selecionadas!')
            else: 
                op2 = list(df.columns)
                op2.insert(0, 'Selecione uma opção')
                
                select_cat_corr = st.selectbox(
                    'Informe a coluna alvo para calcular a correlação com as colunas qualitativas', options=op2)

                cat_corr = {}

                if select_cat_corr != 'Selecione uma opção':
                    for col in cat_features.columns:
                        cat_corr[col] = conditional_entropy(
                            cat_features[col], df[select_cat_corr])

                series_cat_corr = pd.Series(cat_corr, name='correlation')
                st.dataframe(series_cat_corr)

                
                cat_fit_features_add_radio = st.radio('Deseja incluir ou excluir colunas para o pré-processamento?', options=(
                    'Incluir', 'Excluir'
                ), key='1')
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                if cat_fit_features_add_radio in 'Incluir':
                    cat_fit_features_add = st.multiselect(
                    'Selecione as coluna para inclusão', options=list(cat_features.columns))
                    
                    cat_features = cat_features[cat_fit_features_add]

                    if len(cat_fit_features_add) > 1:
                        for col in cat_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    else:
                        if cat_fit_features_add:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=cat_fit_features_add[0], function_operator=dict_db['function_operator'][10], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Colunas selecionadas -> {list(cat_features.columns)}')
                
                elif cat_fit_features_add_radio in 'Excluir':
                    cat_fit_features_delete = st.multiselect(
                        'Seleciona as colunas para exclusão', options=list(cat_features.columns)
                    )
                    cat_features_delete.append(cat_fit_features_delete)
                    
                    cat_features = cat_features.drop(cat_fit_features_delete, axis=1)

                    if len(cat_fit_features_delete) > 1:
                        for col in cat_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][8], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    else:
                        if cat_fit_features_delete:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=cat_fit_features_delete[0], function_operator=dict_db['function_operator'][8], name_operator=dict_db['name_operator'][2], type_operator=dict_db['type_operator'][1], timestamp=datetime.now())

                    
                    st.success(f'Colunas disponíveis -> {list(cat_features.columns)}')
                            
        
        
        # 5 - Feature engineering       
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        st.markdown('### 5 - Transformação de Dados -  Feature Engineering')
        
        op3 = list(num_features.columns)
        op3.insert(0, 'Selecione uma opção')
        
        select_target = st.selectbox('Informe a coluna alvo:', options=list(op3))
        
        st.markdown('<br>', unsafe_allow_html=True)
        
              
        if select_target not in 'Selecione uma opção':
            
            
            if st.checkbox('Colunas quantitativas', key='5'):
                
                is_applied_binning = False
                
                if st.checkbox('Contínuas', key='6'):
                    if st.checkbox('Explicação de Discretização'):        
                        st.markdown(markdown_binning)
            
                    n_bins_slider = st.slider('n_bins', min_value=2, max_value=20, value=5)
                    encode_select = st.selectbox('encode', options=('onehot-dense', 'ordinal'))
                    strategy_select = st.selectbox('strategy', options=('quantile', 'uniform', 'kmeans'))
                    
                    
                    select_col_binning = st.multiselect('Informe as colunas que deseja aplicar a Discretização:', options=list(num_features.drop(select_target, axis=1).columns))
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
                        st.success('Transformação realizada com sucesso!')
                
                if st.checkbox('Discretas e Contínuas'):
                    if st.checkbox('Explicação de Normalização e Padronização'):   
                        st.markdown(markdown_scaling)
                        st.markdown('<br>', unsafe_allow_html=True)
                        st.markdown(markdown_standardization)
                        
                    select_method_var_quantitative = st.selectbox('Escolha o método:', options=('Selecione uma opção', 'Normalização', 'Padronização'))
                    
                    if select_method_var_quantitative in 'Normalização':
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

                            
                            st.success('Transformação realizada com sucesso!')
                        else:
                            num_features = scaling(num_features, select_target)

                            if len(num_features.columns.tolist()) > 1:
                                for col in num_features.columns:
                                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][13], name_operator=dict_db['name_operator'][3], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            else:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][13], name_operator=dict_db['name_operator'][3], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            st.success('Transformação realizada com sucesso!')
                        
                    if select_method_var_quantitative in 'Padronização':
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

                            
                            st.success('Transformação realizada com sucesso!')
                        else:
                            num_features = standardization(num_features, select_target)

                            if len(num_features.columns.tolist()) > 1:
                                for col in num_features.columns:
                                    save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][14], name_operator=dict_db['name_operator'][4], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            else:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=num_features.columns.tolist()[0], function_operator=dict_db['function_operator'][14], name_operator=dict_db['name_operator'][4], type_operator=dict_db['type_operator'][3], timestamp=datetime.now())

                            st.success('Transformação realizada com sucesso!')
            
            st.markdown('<br>', unsafe_allow_html=True)
             
                       
            if st.checkbox('Colunas qualitativas', key='7'):
                
                is_onehot_transform = False
                is_ordinal_transform = False
                
                no_preprocessed = []
                
                if st.checkbox('Nominal (OneHot Encoder)', key='8'):
                    if st.checkbox('Explicação da codificação - OneHot Encoder'):  
                        st.markdown(markdown_onehot)
                        st.markdown('<br>', unsafe_allow_html=True)
                    
                    select_cat_features_nominal = st.multiselect('Informe as colunas que deseja aplicar a codificação - OneHot Encoder:', options=list(cat_features.columns))
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
                        st.success('Transformação realizada com sucesso!')
                
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
            
     
            
        # 8 - Gerar arquivos pré-processado (Traino e teste) / Base Única
        st.markdown('### 6 - Partição de Dados - Treino e Teste ou Base Única')
        
        
        is_select_partition = st.selectbox('Deseja fazer o particionamento do conjunto de dados em Treino e Teste? ', options=('Selecione uma opção', 'Não', 'Sim'))
        st.markdown('<br>', unsafe_allow_html=True)
        
        
        if is_select_partition in 'Selecione uma opção':
            pass
        else:
            if is_select_partition in 'Não':
                op4 = list(num_features.columns)
                op4.insert(0, 'Selecione uma opção')
                
                select_target_partition = st.selectbox('Informe a coluna alvo para realizar o particionamente do conjunto de dados:', options=list(op4))
                            
                
                if select_target_partition is not 'Selecione uma opção':
                    if select_target_partition not in cat_features:
                        X = pd.concat([num_features.drop(select_target_partition, axis=1).reset_index(drop=True), cat_features], axis=1).reset_index(drop=True)
                        
                    y = num_features[select_target_partition].copy()
                    
                    unified_base = pd.concat([X, y], axis=1).reset_index(drop=True)
                    
                    
                    if not unified_base.empty:
                        st.success('Base disponível para download')
                    
                
                    is_completed_select = st.sidebar.selectbox('Finalizou todas as operações de pré-processamento?', options=('Não', 'Sim'))
                    
                    
                    if is_completed_select in 'Sim':
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
                        st.sidebar.warning('Completou o pré-processamento?')
            
            
            
            if is_select_partition in 'Sim':
            
                op5 = list(num_features.columns)
                op5.insert(0, 'Selecione uma opção')
                
                select_target_partition = st.selectbox('Informe a coluna alvo para gerar o conjunto de dados pré-processado:', options=list(op5))
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                
                
                
                if select_target_partition != 'Selecione uma opção':
                    if select_target_partition not in cat_features:
                        X = pd.concat([num_features.drop(select_target_partition, axis=1).reset_index(drop=True), cat_features], axis=1).reset_index(drop=True)
                        
                    y = num_features[select_target_partition].copy()
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    
                    select_test_size = st.slider('Informe qual a proporção do tamanho da base de teste:', min_value=1, max_value=99, value=25)
                    
                    st.write(X.shape)
                    st.write(y.shape)
                        
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(select_test_size) / 100.0)
                    
                    
                    st.write(float(select_test_size) / 100.0)
                    
                    st.markdown('**Base de treino**')
                    st.write(X_train.shape)
                    st.write(y_train.shape)
                    
                    st.markdown('**Base de teste**')
                    st.write(X_test.shape)
                    st.write(y_test.shape)

                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    
                    # 7 - Correção da Amostragem de Dados
                    st.markdown('### 7 - Correção da Amostragem de Dados')
                                      
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    if st.checkbox('Explicação de correção da amostragem de dados'):  
                        st.markdown(markdown_class_desbalance_v2)
                        st.markdown('<br>', unsafe_allow_html=True)
                    
                    if st.checkbox('Explicação do método a utilizar'):
                        st.markdown(markdown_class_desbalance_v3)
                    
                    method_balance_select = st.selectbox('Escolha o método mais apropriado para o seu problema:', options=(
                        'Selecione uma opção', 'Over-sampling', 'Under-sampling'
                    ))
                    
                    if method_balance_select in 'Over-sampling':
                        try:
                            X_train, y_train = over_sampling(X_train, y_train)

                            sampling_cols = X_train.columns.tolist() + [y.name]
                            for col in sampling_cols:
                                save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][17], name_operator=dict_db['name_operator'][9], type_operator=dict_db['type_operator'][2], timestamp=datetime.now())

                            st.success('Over-sampling aplicado com sucesso!')
                        except Exception as e:
                            st.markdown(e)              
                        
                    if method_balance_select in 'Under-sampling':
                        X_train, y_train = under_sampling(X_train, y_train)

                        under_sampling_cols = X_train.columns.tolist() + [y.name]
                        for col in under_sampling_cols:
                            save_to_database_ORM(conn_db, number_workflow=last_number_workflow, name_dataset=str(database_name), name_column=col, function_operator=dict_db['function_operator'][18], name_operator=dict_db['name_operator'][10], type_operator=dict_db['type_operator'][2], timestamp=datetime.now())
                        
                        st.success('Under-sampling aplicado com sucesso!')
                
                
                    train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
                    test = X_test
                
                    is_completed_select = st.sidebar.selectbox('Finalizou todas as operações de pré-processamento?', options=('Não', 'Sim'))
                    
                    
                    if is_completed_select in 'Sim':
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
                        st.sidebar.warning('Completou o pré-processamento?')
                     
        
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
                    st.error('Query inválida!')


        

