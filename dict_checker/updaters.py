"""
    This modeule incapsulate classes for checking dictionaries for update.
    author: i.zemskov
    date: 17.05.2018
"""

import pandas as pd
from aux_functions import compare_str_fields, compare_str_series, to_spreadsheet, excel_date
from sqlalchemy import create_engine
from datetime import date
from day_ahead_scripts import config
import logging
import warnings
import os
import traceback
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class RgeDictUpdater:
    """Class for checking 'dict_gtprge_gen' table for updates."""

    def __init__(self, rio_path, dict_path, begin_date, end_date,
                 version=0, dict_sheet_name='GTPRGE_GEN'):

        """
        Constructor
        :param rio_path: path to the RIO excel file
        :param dict_path: path to the Exergy dict excel file
        :param begin_date: begin date of the checking period datetime.date object
        :param end_date: end date of the checking period datetime.date object
        :param version: version of calculations of model
        :param dict_sheet_name: name of the list with
        """

        self.rio_path = rio_path
        self.dict_path = dict_path
        self.dict_sheet_name = dict_sheet_name
        self.begin_date = begin_date
        self.end_date = end_date
        self.version = version
        self.rio_data, self.dict_data = self.__load_data()
        self.missing_rge_list = self.__load_missing_rge_list()

    def __load_data(self):

        """
        Reads dict and RIO data from the specified file paths
        """

        rio_data = pd.read_excel(self.rio_path)
        dict_data = pd.read_excel(self.dict_path, sheet_name=self.dict_sheet_name)

        # transfer column names to uppercase
        rio_data.columns = [s.upper() for s in rio_data.columns]
        dict_data.columns = [s.upper() for s in dict_data.columns]

        return rio_data, dict_data

    def __load_missing_rge_list(self):

        """
        Finds out which RGEs from the list of RGEs in the result table res_rge of model calculation
         are missing in dict data
        :return: list of missing RGEs (in dict table)
        """

        def launch_query(query):
            """
            Launches specified query on the database
            :param query: string representing sql query
            :return: rows with information concerning missed RGEs
            """
            logging.info('Executing query...')
            engine = create_engine(config.db_conn)
            result = pd.read_sql(query, engine)
            logging.info('Executing finished!')

            return result

        def construct_query():
            """
            Constructs query text
            :return: text of the query
            """

            query = f"""
            select distinct
                rge
            from
                model1.res_rge
            where
                date between '{self.begin_date:%Y-%m-%d}'
                    and '{self.end_date:%Y-%m-%d}'
                and version = {self.version}
            except
            select distinct
                rge
            from
                model1.dict_gtprge_gen
            where
                dict_version in (
                    select 
                        max(dict_version) 
                    from 
                        model1.dict_gtprge_gen
                    where
                       date between '{self.begin_date:%Y-%m-%d}'
                            and '{self.end_date:%Y-%m-%d}'
                        and version = {self.version})
            order by 
                rge"""

            return query

        # list of missing rges from database (res_rge table)
        list_from_db = set(launch_query(construct_query())['rge'].values)

        return list(list_from_db)

    def compare_gtprge_rio(self, left_on, right_on, comp_cols, draw_map=False):

        """
        Compares rows for each rge in dict with rge rows in RIO and finds mismatches in values of fields
            for the columns ['STATION_CODE', 'GTP_CODE', 'STATION_NAME', 'GTP_NAME', 'RGE_NAME']
        :param left_on: list of columns of 'left' table in merge
        :param right_on: list of columns of 'right' table in merge
        :param comp_cols: list of columns to compare values of
        :param draw_map: boolean, True - draw heatmap of mismatches, False - don't draw
        :return: dictionary with merged (dict + RIO) dataframe and various masks of matching:
                    result_mask_dict - mask of matchings by compared columns,
                    matched_rows_mask - mask of rows with total matching by compared columns
                    unmatched_rows_mask - logical NOT, applied to matched_rows_mask
        """

        # select only actual information
        actual_dict_rows = self.dict_data[self.dict_data.DATE_TO.isnull()]

        # merge two datasets
        dict_rio = actual_dict_rows.merge(right=self.rio_data, left_on=left_on, right_on=right_on,
                                          how='left', suffixes=('_DICT', '_RIO')).sort_values(by=left_on)
        dict_rio = dict_rio.drop(right_on, axis=1)

        # select rows, that weren't found in rio_data
        not_found_in_rio = dict_rio[dict_rio['STATION_CODE_RIO'].isna()]

        # log number of not-found rows
        logging.info(f'{not_found_in_rio.shape[0]} RGEs were not found in RIO')

        # select only rows with STATION_CODE, that is found in rio_data
        dict_rio = dict_rio[dict_rio['STATION_CODE_RIO'].notna()]

        # check if merging was OK
        assert (actual_dict_rows.shape[0] == dict_rio.shape[0] + not_found_in_rio.shape[0])

        # check for duplicate RGE codes
        assert (len(pd.unique(dict_rio.RGE_NUM)) == dict_rio.shape[0])

        # obtaining map of unmatched fields
        result_mask = []
        result_mask_dict = {}
        for col in comp_cols:
            col1 = col + '_DICT'
            col2 = col + '_RIO'
            # logging.info('Comparing columns ' + col1 + ' and ' + col2)
            result_mask.append(list(compare_str_series(dict_rio[col1], dict_rio[col2])))
            result_mask_dict[col] = list(compare_str_series(dict_rio[col1], dict_rio[col2]))

        result_mask = np.array(result_mask)

        # logical sum: checking if there's any fully matched rows
        matched_rows_mask = (np.sum(result_mask, axis=0) == result_mask.shape[0])
        unmatched_rows_mask = (np.sum(result_mask, axis=0) != result_mask.shape[0])

        try:
            assert (np.sum(unmatched_rows_mask) + np.sum(matched_rows_mask) == result_mask.shape[1])
        except AssertionError:
            logging.info('Sum of matched and unmatched rows is not equal to the total number of rows compared!')
            traceback.print_exc()

        # plotting report of matched/unmatched rows
        if draw_map:
            fig, ax = plt.subplots(figsize=(8, 12))
            sns.heatmap(result_mask.astype(int).T)
            plt.title(f'Map of matched/unmatched columns for {self.dict_sheet_name} dictionary')
            ax.set_xticklabels(comp_cols)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'C:\\temp\\unmatching_map_{self.begin_date}.png')

        result_dict = {'result_mask_dict': result_mask_dict,
                       'matched_rows_mask': matched_rows_mask,
                       'unmatched_rows_mask': unmatched_rows_mask,
                       'dict_rio': dict_rio,
                       'not_found_in_rio': not_found_in_rio}

        return result_dict

    def update_gtprge(self, df, nf_df, by_col_mask, matched_mask, author):

        """
        Updates the dict_gtprge_gen dictionary with respect to RIO data and new RGEs from res_rge
            table from RSV imitator database, scheme model1. Uses results from the compare_gtprge_rio method.
        :param df: output from compare_gtprge_rio(), result of the merge of dict_data and rio_data.
        :param nf_df: rows in dict_data, that weren't found in rio
        :param by_col_mask: mask of matchings by compared columns from compare_gtprge_rio()
        :param matched_mask: mask of rows with total matching by compared columns from compare_gtprge_rio()
        :param author: str with author value to write to the updated dictionary
        :return: updated dictionary dict_gtprge_gen
        """

        def generate_comment(row):

            """Generate text for 'comment' field, based on unmatching condition.
                    :param: row: dataframe row, containing station code and gtp_code variables
                        from both dict and rio data.
                    :return: text of the comment field
            """

            if not compare_str_fields(row[0], row[1]) and not compare_str_fields(row[2], row[3]):
                return f'{row[0]}, {row[2]}'
            elif not compare_str_fields(row[0], row[1]) and compare_str_fields(row[2], row[3]):
                return f'{row[0]}'
            elif compare_str_fields(row[0], row[1]) and not compare_str_fields(row[2], row[3]):
                return f'{row[2]}'
            else:
                return ''

        out_cols = ['STATION_CODE', 'STATION_NAME', 'GTP_CODE',
                    'GTP_NAME', 'RGE_NUM', 'RGE_NAME', 'DAY_AHEAD_TYPE',
                    'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']

        src_cols = ['STATION_CODE_DICT', 'STATION_NAME_DICT', 'GTP_CODE_DICT',
                    'GTP_NAME_DICT', 'RGE_NUM', 'RGE_NAME_DICT', 'DAY_AHEAD_TYPE',
                    'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']

        # condition 1
        unmatched_codes_rows = df.iloc[np.logical_not(np.array(by_col_mask.get('STATION_CODE'))) |
                                       np.logical_not(np.array(by_col_mask.get('GTP_CODE')))]

        # condition 2
        unmatched_names_rows = df.iloc[np.logical_not(np.array(by_col_mask.get('STATION_NAME'))) &
                                       np.array(by_col_mask.get('STATION_CODE')) &
                                       np.array(by_col_mask.get('GTP_CODE'))]

        # condition 3
        matched_rows = df.iloc[matched_mask]

        # other rows
        other_rows = df.iloc[np.array(by_col_mask.get('STATION_CODE')) &
                             np.array(by_col_mask.get('GTP_CODE'))
                             & np.array(by_col_mask.get('STATION_NAME')) & np.logical_not(matched_mask)]

        # rows, that weren't found in rio_data
        not_found_rows = nf_df[src_cols]
        not_found_rows.columns = out_cols

        num_rows = unmatched_codes_rows.shape[0] + unmatched_names_rows.shape[0] \
                   + matched_rows.shape[0] + other_rows.shape[0]

        assert (num_rows == df.shape[0])

        # manage with 1st-condition rows if exist
        if not unmatched_codes_rows.empty:
            old_rows = unmatched_codes_rows[['STATION_CODE_DICT', 'STATION_NAME_DICT', 'GTP_CODE_DICT',
                                             'GTP_NAME_DICT', 'RGE_NUM', 'RGE_NAME_DICT', 'DAY_AHEAD_TYPE',
                                             'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']]

            # update value of the date_to field
            old_rows['DATE_TO'] = pd.to_datetime(date.today().replace(day=1).isoformat())

            new_rows = unmatched_codes_rows[['STATION_CODE_RIO', 'STATION_NAME_RIO', 'GTP_CODE_RIO',
                                             'GTP_NAME_RIO', 'RGE_NUM', 'RGE_NAME_RIO', 'DAY_AHEAD_TYPE',
                                             'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']]

            # update value of the day_ahead_type field
            new_rows['DAY_AHEAD_TYPE'] = unmatched_codes_rows['IS_SPOT_TRADER'].map(lambda v: 0 if v == 1 else 2)
            new_rows['DATE_FROM'] = pd.to_datetime(date.today().replace(day=1).isoformat())
            new_rows['DATE_TO'] = np.NaN
            new_rows['AUTHOR'] = author
            new_rows['COMMENT'] = unmatched_codes_rows[['STATION_CODE_DICT', 'STATION_CODE_RIO',
                                                        'GTP_CODE_DICT', 'GTP_CODE_RIO']] \
                .apply(generate_comment, axis=1)
            old_rows.columns = out_cols
            new_rows.columns = out_cols
            unmatched_codes_fixed = pd.concat([old_rows, new_rows], axis=0)
            unmatched_codes_fixed.columns = out_cols
            logging.info(f'{unmatched_codes_fixed.shape[0] - unmatched_codes_rows.shape[0]} rows were added!')

            # deleting rows with synthetic data and not empty DATE_TO
            synths_removed = unmatched_codes_fixed[(unmatched_codes_fixed.STATION_CODE == 'GKRAOEES')
                                                   & (unmatched_codes_fixed.DATE_TO.notna())].shape[0]
            unmatched_codes_fixed = unmatched_codes_fixed[np.logical_not(
                (unmatched_codes_fixed.STATION_CODE == 'GKRAOEES') & (unmatched_codes_fixed.DATE_TO.notna()))]

            logging.info(f'{synths_removed} rows with old synthetic data were removed afterwards.')

        else:

            unmatched_codes_fixed = pd.DataFrame(columns=out_cols)

        # manage with 2nd-condition rows if exist: take station_name from rio column
        if not unmatched_names_rows.empty:
            unmatched_names_fixed = unmatched_names_rows[['STATION_CODE_DICT', 'STATION_NAME_RIO', 'GTP_CODE_DICT',
                                                          'GTP_NAME_DICT', 'RGE_NUM', 'RGE_NAME_DICT', 'DAY_AHEAD_TYPE',
                                                          'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED',
                                                          'COMMENT']]
            unmatched_names_fixed.columns = out_cols
        else:
            unmatched_names_fixed = pd.DataFrame(columns=out_cols)

        # finally, manage with matched rows
        matched_rows = matched_rows[src_cols]
        matched_rows.columns = out_cols

        # manage with other rows
        other_rows = other_rows[src_cols]
        other_rows.columns = out_cols

        # insertion of data concerning missing RGEs; in case of unknown info - synthetic value into all columns
        missing_rge_data = self.rio_data[self.rio_data.RGE_CODE.isin(self.missing_rge_list)]
        dict_rge_cols = self.dict_data.columns
        missing_rge_data = missing_rge_data.rename(index=str, columns={'RGE_CODE': 'RGE_NUM'})
        missing_rge_data['DAY_AHEAD_TYPE'] = missing_rge_data['IS_SPOT_TRADER'].map(lambda v: 0 if v == 1 else 2)
        missing_rge_data['DATE_FROM'] = pd.to_datetime(date.today().replace(day=1).isoformat())
        missing_rge_data['DATE_TO'] = np.NaN
        missing_rge_data['AUTHOR'] = author
        missing_rge_data['STATE'] = 3
        missing_rge_data['DATE_MODIFIED'] = pd.to_datetime(date.today().replace(day=1).isoformat())
        missing_rge_data['COMMENT'] = ''
        # select only columns which a presented in dict_data
        missing_rge_data = missing_rge_data[[c for c in missing_rge_data.columns
                                             if c in dict_rge_cols]]

        # determine which of missing RGEs still not added in dict_data
        still_missing_rges = [rge for rge in self.missing_rge_list
                              if rge not in list(pd.unique(missing_rge_data.RGE_NUM))]

        # construct synthetic data for such RGEs (this algorithm is provided by Sergey)

        if still_missing_rges:

            # create empty data frame with index as still_missing_rges list
            still_missing_data = pd.DataFrame(columns=missing_rge_data.columns.remove('RGE_NUM'),
                                              index=still_missing_rges)
            # filling with synthetic data
            for rge in still_missing_rges:
                still_missing_data.loc[rge] = pd.Series({'STATION_CODE': 'GKRAOEES',
                                                         'STATION_NAME': 'GKRAOEES',
                                                         'GTP_CODE': 'GKRAOEES',
                                                         'GTP_NAME': 'GKRAOEES',
                                                         'RGE_NAME': 'GKRAOEES',
                                                         'DAY_AHEAD_TYPE': 2,
                                                         'DATE_FROM': pd.to_datetime(date.today()
                                                                                     .replace(day=1).isoformat()),
                                                         'DATE_TO': np.NaN,
                                                         'AUTHOR': author,
                                                         'STATE': 3,
                                                         'COMMENT': ''})
                still_missing_data = still_missing_data.reset_index()
                still_missing_data.columns = ['RGE_NUM'].extend(missing_rge_data.columns.remove('RGE_NUM'))
            missing_rge_data = pd.concat([missing_rge_data, still_missing_data.reset_index()], axis=0)

        # set column names to lowercase for compatibility with Exergy loader
        # for some reason order of columns in resulting dataframe differs from source dataframes
        # managed to find some info of this bug in pandas
        result_df = pd.concat([matched_rows, unmatched_codes_fixed, unmatched_names_fixed,
                               other_rows, not_found_rows, missing_rge_data], axis=0) \
            .sort_values(['RGE_NUM', 'DATE_TO'])
        # reorder columns back
        result_df = result_df[out_cols]

        # select archived data from dict_gtprge_gen (by DATE_TO.notna() condition)
        archived_dict_rows = self.dict_data[self.dict_data.DATE_TO.notna()]
        updated_part = result_df.shape[0]
        result_df = pd.concat([result_df, archived_dict_rows], axis=0)
        cols_lower = [c.lower() for c in out_cols]
        result_df.columns = cols_lower

        result_df['date_from'] = result_df['date_from'].map(excel_date)
        result_df['date_to'] = result_df['date_to'].map(excel_date)
        result_df['date_modified'] = result_df['date_modified'].map(excel_date)

        # check for a constant number of columns
        assert (self.dict_data.shape[1] == result_df.shape[1])

        # check if sum of numbers of rows in different components of result_df is equal to
        # number of columns of result_df
        # assert (result_df.shape[0] == not_found_rows.shape[0] + matched_rows.shape[0] + new_rows.shape[0]
        #        + old_rows.shape[0] + archived_dict_rows.shape[0] + unmatched_names_fixed.shape[0]
        #        + other_rows.shape[0])

        logging.info(f'\nInitial number of rows in dict_gtprge_gen: {self.dict_data.shape[0]} \n'
                     f'Initial number of rows in rio_data: {self.rio_data.shape[0]} \n'
                     f'{matched_rows.shape[0]} fully matched rows were found. \n'
                     f'{unmatched_codes_fixed.shape[0] + missing_rge_data.shape[0]} new rows were added. \n'
                     f'{unmatched_codes_fixed.shape[0]} of them are with updated params. \n'
                     f'{missing_rge_data.shape[0]} of them are with new RGEs. \n'
                     f'In {unmatched_names_fixed.shape[0]} of rows station_name was updated. \n'
                     f'In {other_rows.shape[0]} some differences were found, but were ignored determinately. \n'
                     f'{archived_dict_rows.shape[0]} rows were initially not actual in dict_data.. \n'
                     f'List of newly added RGEs, that are not found in RIO data: {still_missing_rges} \n'
                     f'Number of rows in updated part of dict_gtprge_gen: {updated_part} \n'
                     f'Total number of rows in updated dict_gtprge_gen: {result_df.shape[0]}\n')

        return result_df


class RegistryGenUpdater:
    """Class for checking 'dict_registry_gen' table for updates."""

    def __init__(self, rio_path, dict_path, begin_date, end_date, version=0, dict_sheet_name='REGISTRY_GEN'):
        self.rio_path = rio_path
        self.dict_path = dict_path
        self.begin_date = begin_date
        self.end_date = end_date
        self.dict_sheet_name = dict_sheet_name
        self.version = version
        self.rio_data, self.dict_data = self.__load_data()

    def __load_data(self):

        """
        Reads dict and RIO data from the specified file paths
        """

        rio_data = pd.read_excel(self.rio_path)
        rio_data = rio_data[['TRADER_CODE', 'COMPANY_NAME', 'STATION_CODE', 'STATION_NAME']] \
            .rename(columns={'COMPANY_NAME': 'TRADER_NAME'}) \
            .drop_duplicates()
        dict_data = pd.read_excel(io=self.dict_path, sheet_name=self.dict_sheet_name)

        # transfer column names to uppercase
        rio_data.columns = [s.upper() for s in rio_data.columns]
        dict_data.columns = [s.upper() for s in dict_data.columns]

        return rio_data, dict_data

    def compare_registry_gen(self, left_on, right_on, comp_cols, draw_map=False):

        """
        Compares rows for each rge in dict with rge rows in RIO and finds mismatches in values of fields
            for the columns ['STATION_NAME', 'TRADER_CODE', 'TRADER_NAME']
        :param left_on:
        :param right_on:
        :param comp_cols:
        :param draw_map:
        :return: dictionary with merged (dict + RIO) dataframe and various masks of matching:
                    result_mask_dict - mask of matchings by compared columns,
                    matched_rows_mask - mask of rows with total matching by compared columns
                    unmatched_rows_mask - logical NOT, applied to matched_rows_mask
                    not_found_in_rio - rows in dict_data, that weren't found in rio
        """

        # select only actual information
        actual_dict_rows = self.dict_data[self.dict_data.DATE_TO.isnull()]

        # merge two datasets
        dict_rio = actual_dict_rows.merge(right=self.rio_data, left_on=left_on, right_on=right_on,
                                          how='left', suffixes=('_DICT', '_RIO')).sort_values(by=left_on)

        # select rows, that weren't found in rio_data
        not_found_in_rio = dict_rio[dict_rio['TRADER_CODE_RIO'].isna()]

        # select only rows with STATION_CODE, that is found in rio_data
        dict_rio = dict_rio[dict_rio['TRADER_CODE_RIO'].notna()]

        # check if merging was OK
        assert (actual_dict_rows.shape[0] == dict_rio.shape[0] + not_found_in_rio.shape[0])

        # check for duplicate station codes
        assert (len(pd.unique(dict_rio.STATION_CODE)) == dict_rio.shape[0])

        # obtaining map of unmatched fields
        result_mask = []
        result_mask_dict = {}
        for col in comp_cols:
            col1 = col + '_DICT'
            col2 = col + '_RIO'
            # logging.info('Comparing columns ' + col1 + ' and ' + col2)
            result_mask.append(list(compare_str_series(dict_rio[col1], dict_rio[col2])))
            result_mask_dict[col] = list(compare_str_series(dict_rio[col1], dict_rio[col2]))

        result_mask = np.array(result_mask)

        # logical sum: checking if there's any fully matched rows
        matched_rows_mask = (np.sum(result_mask, axis=0) == result_mask.shape[0])
        unmatched_rows_mask = (np.sum(result_mask, axis=0) != result_mask.shape[0])

        try:
            assert (np.sum(unmatched_rows_mask) + np.sum(matched_rows_mask) == result_mask.shape[1])
            assert (np.sum(unmatched_rows_mask) == np.sum(np.logical_not(np.array(result_mask_dict['STATION_NAME'])) |
                                                          np.logical_not(np.array(result_mask_dict['TRADER_CODE'])) |
                                                          np.logical_not(np.array(result_mask_dict['TRADER_NAME']))))
            assert (dict_rio.shape[0] == np.sum(unmatched_rows_mask) + np.sum(matched_rows_mask))

        except AssertionError:
            logging.info('Sum of matched and unmatched rows is not equal to the total number of rows compared!')
            traceback.print_exc()

        if draw_map:
            fig, ax = plt.subplots(figsize=(8, 12))
            sns.heatmap(result_mask.astype(int).T)
            plt.title(f'Map of matched/unmatched columns for {self.dict_sheet_name} dictionary')
            ax.set_xticklabels(comp_cols)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'C:\\temp\\unmatching_map_{self.begin_date}.png')

        result = {'result_mask_dict': result_mask_dict,
                  'matched_rows_mask': matched_rows_mask,
                  'unmatched_rows_mask': unmatched_rows_mask,
                  'dict_rio': dict_rio,
                  'not_found_in_rio': not_found_in_rio}

        return result

    def update_registry_gen(self, df, nf_df, by_col_mask, matched_mask, author):

        """
        Updates information of REGISTRY_GEN dictionary
        :param df: output from compare_registry_gen(), result of the merge of dict_data and rio_data
        :param nf_df: rows in dict_data, that weren't found in rio
        :param by_col_mask: mask of matchings by compared columns from compare_gtprge_rio()
        :param matched_mask: mask of rows with total matching by compared columns from compare_gtprge_rio()
        :param author: str with author value to write to the updated dictionary
        :return: updated dictionary dict_registry_gen
        """

        def retrieve_holding_info(df, miss):

            """
            Function for retrieving information about holding from
                existing pairs trader_code -> holding in dict_registry_gen
            :param df: distinct selection of pairs tradec_code -> holding
                        from dict_registry_gen table (columns: TRADER_CODE, HOLDING, CODE)
            :param miss: list of trader_code values of dataframe
                                with missing stations
            :return: dict of structure: {'TRADER_CODE': ['HOLDING', 'CODE']}, where CODE - holding code,
                HOLDING - holding name
            """

            res = dict()
            for k, v in miss.items():
                df_t = df[df.TRADER_CODE == k].drop_duplicates()
                if df_t.empty:
                    res[k] = (v, k)
                    logging.info(f'Trader code {k} was not mentioned before in dict_registry_gen table!')
                else:
                    res[k] = (df_t.HOLDING.values[0], df_t.CODE.values[0])
            return res

        def extract_station_type(st_name):

            """Function for generating station_type value from station_name
                Attributes: st_name - string with station_name value

                :return: station type code"""

            s_u = st_name.upper()
            if any(word in s_u for word in ['ТЭЦ', 'ТЭС', 'ГПЭС', 'ДЭС', 'ПГУ']):
                return 1
            elif 'ГЭС' in s_u:
                return 2
            elif 'АЭС' in s_u:
                return 3
            elif 'ГРЭС' in s_u:
                return 4
            elif 'СЭС' in s_u or 'СОЛНЕЧНАЯ' in s_u:
                return 5
            elif 'ВЭС' in s_u or 'ВЕТРОВАЯ' in s_u:
                return 6
            else:
                logging.info(f"Тип станции '{st_name}' не опрелелен")
                return 0

        out_cols = ['TRADER_CODE', 'TRADER_NAME', 'STATION_CODE',
                    'STATION_NAME', 'STATION_TYPE', 'HOLDING', 'CODE',
                    'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']

        src_cols = ['TRADER_CODE_DICT', 'TRADER_NAME_DICT', 'STATION_CODE',
                    'STATION_NAME_DICT', 'STATION_TYPE', 'HOLDING', 'CODE',
                    'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']

        # condition 1
        unmatched_codes_rows = df.iloc[np.logical_not(np.array(by_col_mask.get('TRADER_CODE')))]

        # condition 2
        unmatched_names_rows = df.iloc[(np.logical_not(by_col_mask.get('STATION_NAME')) |
                                        np.logical_not(by_col_mask.get('TRADER_NAME'))) &
                                       np.array(by_col_mask.get('TRADER_CODE'))]

        # condition 3
        matched_rows = df.iloc[matched_mask]

        # other rows
        other_rows = df.iloc[np.array(by_col_mask.get('TRADER_CODE')) &
                             np.array(by_col_mask.get('TRADER_NAME')) &
                             np.array(by_col_mask.get('STATION_NAME')) & np.logical_not(matched_mask)]

        num_rows = unmatched_codes_rows.shape[0] + unmatched_names_rows.shape[0] \
                   + matched_rows.shape[0] + other_rows.shape[0]

        # rows, that weren't found in rio_data
        not_found_rows = nf_df[src_cols]
        not_found_rows.columns = out_cols

        # check if something is wrong with logic of comparison vectors
        assert (num_rows == df.shape[0])

        # manage with 1st-condition rows if exist
        if not unmatched_codes_rows.empty:
            old_rows = unmatched_codes_rows[['TRADER_CODE_DICT', 'TRADER_NAME_DICT', 'STATION_CODE',
                                             'STATION_NAME_DICT', 'STATION_TYPE', 'HOLDING', 'CODE',
                                             'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']]

            # update value of the date_to field
            old_rows['DATE_TO'] = pd.to_datetime(date.today().replace(day=1).isoformat())

            new_rows = unmatched_codes_rows[['TRADER_CODE_RIO', 'TRADER_NAME_RIO', 'STATION_CODE',
                                             'STATION_NAME_RIO', 'STATION_TYPE', 'HOLDING', 'CODE',
                                             'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED', 'COMMENT']]

            # update holding fields with respect to new trader_code and trader_name
            miss = new_rows[['TRADER_CODE_RIO', 'TRADER_NAME_RIO']].drop_duplicates().set_index('TRADER_CODE_RIO'). \
                to_dict().get('TRADER_NAME_RIO')
            holding_tcode_dict = retrieve_holding_info(self.dict_data[['TRADER_CODE', 'TRADER_NAME', 'HOLDING', 'CODE']]
                                                       .drop_duplicates(), miss)
            logging.info('Newly found holdings: \n' + str(holding_tcode_dict))
            new_rows['CODE'] = new_rows.TRADER_CODE_RIO.map(lambda t: holding_tcode_dict.get(t)[1])
            new_rows['HOLDING'] = new_rows.TRADER_CODE_RIO.map(lambda t: holding_tcode_dict.get(t)[0])

            # update value of the day_ahead_type field
            new_rows['DATE_FROM'] = pd.to_datetime(date.today().replace(day=1).isoformat())
            new_rows['DATE_TO'] = np.NaN
            new_rows['AUTHOR'] = author

            # generate comments by changes in specified fields
            new_rows['COMMENT'] = unmatched_codes_rows[['TRADER_CODE_DICT']]

            old_rows.columns = out_cols
            new_rows.columns = out_cols

            # check that number of rows in old_rows and new_rows
            assert (old_rows.shape[0] == new_rows.shape[0])

            unmatched_codes_fixed = pd.concat([old_rows, new_rows], axis=0)
            # unmatched_codes_fixed.columns = out_cols
            logging.info(f'{unmatched_codes_fixed.shape[0] - unmatched_codes_rows.shape[0]} rows were added!')

        else:

            unmatched_codes_fixed = pd.DataFrame(columns=out_cols)

        # manage with 2nd-condition rows if exist: take station_name from rio column
        if not unmatched_names_rows.empty:
            unmatched_names_fixed = unmatched_names_rows[['TRADER_CODE_RIO', 'TRADER_NAME_RIO', 'STATION_CODE',
                                                          'STATION_NAME_RIO', 'STATION_TYPE', 'HOLDING', 'CODE',
                                                          'DATE_FROM', 'DATE_TO', 'AUTHOR', 'STATE', 'DATE_MODIFIED',
                                                          'COMMENT']]
            unmatched_names_fixed.columns = out_cols
        else:
            unmatched_names_fixed = pd.DataFrame(columns=out_cols)

        # finally, manage with matched rows
        matched_rows = matched_rows[src_cols]
        matched_rows.columns = out_cols

        # manage with other rows
        other_rows = other_rows[src_cols]
        other_rows.columns = out_cols

        rio_stations = list(pd.unique(self.rio_data.STATION_CODE))
        dict_stations = list(pd.unique(self.dict_data.STATION_CODE))
        missing_stations = [st for st in rio_stations if st not in dict_stations]
        if missing_stations:
            # data from rio for the missing (in dict_data) stations
            new_stations_fixed = self.rio_data[self.rio_data.STATION_CODE.isin(missing_stations)] \
                .filter(items=['TRADER_CODE', 'TRADER_NAME', 'STATION_CODE', 'STATION_NAME'], axis=1) \
                .drop_duplicates()
            # generate STATION_TYPE variable
            new_stations_fixed['STATION_TYPE'] = new_stations_fixed.STATION_NAME.map(extract_station_type)
            miss = new_stations_fixed[['TRADER_CODE', 'TRADER_NAME']].drop_duplicates().set_index('TRADER_CODE'). \
                to_dict().get('TRADER_NAME')
            holding_tcode_dict = retrieve_holding_info(self.dict_data[['TRADER_CODE', 'TRADER_NAME', 'HOLDING', 'CODE']]
                                                       .drop_duplicates(), miss)
            new_stations_fixed['HOLDING'] = new_stations_fixed.TRADER_CODE.map(lambda t: holding_tcode_dict.get(t)[0])
            new_stations_fixed['CODE'] = new_stations_fixed.TRADER_CODE.map(lambda t: holding_tcode_dict.get(t)[1])
            new_stations_fixed['DATE_FROM'] = pd.to_datetime(date.today().replace(day=1).isoformat())
            new_stations_fixed['DATE_TO'] = np.NaN
            new_stations_fixed['AUTHOR'] = author

        else:
            new_stations_fixed = pd.DataFrame(columns=out_cols)

        # set column names to lowercase for compatibility with Exergy loader
        result_df = pd.concat([matched_rows, unmatched_codes_fixed, unmatched_names_fixed,
                               other_rows, not_found_rows], axis=0) \
            .sort_values(['STATION_CODE', 'DATE_TO'])

        # select archived data from dict_gtprge_gen (by DATE_TO.notna() condition)
        archived_dict_rows = self.dict_data[self.dict_data.DATE_TO.notna()]
        not_actual_rows = archived_dict_rows.shape[0]
        updated_part_dim = result_df.shape[0]
        result_df = pd.concat([result_df, archived_dict_rows], axis=0).sort_values(['STATION_CODE', 'DATE_TO'])
        cols_lower = [c.lower() for c in out_cols]
        result_df.columns = cols_lower
        result_df['date_from'] = result_df['date_from'].map(excel_date)
        result_df['date_to'] = result_df['date_to'].map(excel_date)
        result_df['date_modified'] = result_df['date_modified'].map(excel_date)

        # check for a constant number of columns
        assert (self.dict_data.shape[1] == result_df.shape[1])

        # check if sum of numbers of rows in different components of result_df is equal to
        # number of columns of result_df
        assert (result_df.shape[0] == not_found_rows.shape[0] + matched_rows.shape[0] + new_rows.shape[0]
                + old_rows.shape[0] + new_stations_fixed.shape[0] + unmatched_names_fixed.shape[0]
                + other_rows.shape[0] + not_actual_rows)

        logging.info(f'\nInitial number of rows in dict_registry_gen: {self.dict_data.shape[0]} \n'
                     f'Initial number of rows in rio_data: {self.rio_data.shape[0]} \n'
                     f'{not_found_rows.shape[0]} rows of dict_data were not found in rio_data \n'
                     f'{matched_rows.shape[0]} fully matched rows were found. \n'
                     f'{new_rows.shape[0] + new_stations_fixed.shape[0]} new rows were added. \n'
                     f'{new_rows.shape[0]} of them are with updated params. \n'
                     f'{new_stations_fixed.shape[0]} of them are with new stations. \n'
                     f'In {unmatched_names_fixed.shape[0]} of rows station_name or trader_name were updated. \n'
                     f'In {other_rows.shape[0]} some differences were found, but were ignored determinately. \n'
                     f'{not_actual_rows} rows were initially not actual in dict_data. \n'
                     f'List of newly added stations, that are not found in RIO data: {missing_stations} \n'
                     f'Updated part of dict_data has {updated_part_dim} rows. \n'
                     f'Total number of rows in updated dict_registry_gen: {result_df.shape[0]}\n')

        return result_df


class HoldingsUpdater:
    """ Encapsulates functions for updating holding information of dictionary """

    def __init__(self, dict_path, registry_sheet, holdings_sheet):

        """
        Constructor
        :param dict_path: path to the dictionary excel file
        :param registry_sheet: name of the sheet with registry_gen data
        :param holdings_sheet: name of sheet with holdings data
        """

        self.dict_path = dict_path
        self.registry_sheet = registry_sheet
        self.holdings_sheet = holdings_sheet
        self.registry_data, self.holdings_data = self.__load_data()

    def __load_data(self):

        """
        Loads data from the specified file paths
        :return: pandas data frames with loaded data
        """

        registry_data = pd.read_excel(self.dict_path, sheet_name=self.registry_sheet)
        holdings_data = pd.read_excel(self.dict_path, sheet_name=self.holdings_sheet)
        self.init_cols = holdings_data.columns

        # transfer column names to uppercase
        registry_data.columns = [s.upper() for s in registry_data.columns]
        holdings_data.columns = [s.upper() for s in holdings_data.columns]

        return registry_data, holdings_data

    def update_holdings_info(self, author):

        """
        Updates data of holdings from dictionary
        :param author: str with author of changes name
        :return: updated data
        """

        logging.info('Start updating holdings info...')

        # select only actual rows
        registry_data = self.registry_data[self.registry_data.DATE_TO.isna()]
        holdings_data = self.holdings_data[self.holdings_data.DATE_TO.isna()]

        reg_holdings = registry_data[['HOLDING', 'CODE']].drop_duplicates()
        hold_holdings = holdings_data[['HOLDING', 'CODE']].drop_duplicates()

        reg_hold_holdings = reg_holdings.merge(right=hold_holdings, on='CODE', how='left',
                                               suffixes=('_REG', '_HOLD'))

        # select all new rows
        holdings_new = reg_hold_holdings[reg_hold_holdings.isnull().any(axis=1)][['HOLDING_REG', 'CODE']] \
            .rename(index=str, columns={'HOLDING_REG': 'HOLDING'})

        if holdings_new.empty:
            holdings_new = pd.DataFrame(columns=[c.upper() for c in self.init_cols])
        else:
            holdings_new['DATE_FROM'] = pd.to_datetime(date.today().replace(day=1).isoformat())
            holdings_new['DATE_TO'] = np.NaN
            holdings_new['AUTHOR'] = author
            holdings_new['STATE'] = 3
            holdings_new['DATE_MODIFIED'] = pd.to_datetime(date.today().replace(day=1).isoformat())
            holdings_new['COMMENT'] = ''

        result_holdings = pd.concat([holdings_data, holdings_new], axis=0)

        # check for equality in column dimension
        assert (result_holdings.shape[1] == holdings_data.shape[1])

        result_holdings = result_holdings[[c.upper() for c in self.init_cols]]
        result_holdings.columns = self.init_cols
        result_holdings = result_holdings.sort_values(['holding'])

        logging.info(f'{self.holdings_data.shape[0]} rows were in initial data. \n'
                     f'{hold_holdings.shape[0]} actual rows were in initial data. \n'
                     f'{holdings_new.shape[0]} new holding rows were added. \n'
                     f'{result_holdings.shape[0]} are in updated holdings data. \n')

        return result_holdings
