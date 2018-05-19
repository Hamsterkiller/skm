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
                date between '{self.begin_date:%Y-%m-%d}' 
                    and '{self.end_date:%Y-%m-%d}'
            order by 
                rge"""

            return query

        return list(set(launch_query(construct_query())['rge'].values))

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

        # check if merging was OK
        assert (actual_dict_rows.shape[0] == dict_rio.shape[0])

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
            plt.title('Map of matched/unmatched columns')
            ax.set_xticklabels(comp_cols)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'C:\\temp\\unmatching_map_{self.begin_date}.png')

        result = {'result_mask_dict': result_mask_dict,
                  'matched_rows_mask': matched_rows_mask,
                  'unmatched_rows_mask': unmatched_rows_mask,
                  'dict_rio': dict_rio}

        return result

    def update_gtprge(self, df, by_col_mask, matched_mask, author):

        """
        Updates the dict_gtprge_gen dictionary with respect to RIO data and new RGEs from res_rge
            table from RSV imitator database, scheme model1. Uses results from the compare_gtprge_rio method.
        :param df: output from compare_gtprge_rio(), result of the merge of dict_data and rio_data.
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
        unmatched_names_rows = df.iloc[np.logical_not(by_col_mask.get('STATION_NAME')) &
                                       np.array(by_col_mask.get('STATION_CODE')) &
                                       np.array(by_col_mask.get('GTP_CODE'))]

        # condition 3
        matched_rows = df.iloc[matched_mask]

        # other rows
        other_rows = df.iloc[np.array(by_col_mask.get('STATION_CODE')) &
                             np.array(by_col_mask.get('GTP_CODE'))
                             & by_col_mask.get('STATION_NAME') & np.logical_not(matched_mask)]

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
        missing_rge_data['COMMENT'] = ''
        # select only columns which a presented in dict_data
        missing_rge_data = missing_rge_data[[c for c in missing_rge_data.columns
                                             if c in dict_rge_cols]]

        # determine which of missing RGEs still not added in dict_data
        still_missing_rges = [rge for rge in self.missing_rge_list
                              if rge not in list(pd.unique(missing_rge_data.RGE_NUM))]

        # construct synthetic data for such RGEs (this algorithm is provided by Sergey)

        if still_missing_rges:
            still_missing_data = pd.DataFrame(columns=missing_rge_data.columns.remove('RGE_NUM'),
                                              index=still_missing_rges)
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
        result_df = pd.concat([matched_rows, unmatched_codes_fixed, unmatched_names_fixed, other_rows], axis=0) \
            .sort_values(['RGE_NUM', 'DATE_TO'])

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

        logging.info(f'\nInitial number of rows in dict_gtprge_gen: {self.dict_data.shape[0]} \n'
                     f'Initial number of rows in rio_data: {self.rio_data.shape[0]} \n'
                     f'{matched_rows.shape[0]} fully matched rows were found. \n'
                     f'{unmatched_codes_fixed.shape[0] + missing_rge_data.shape[0]} new rows were added. \n'
                     f'{unmatched_codes_fixed.shape[0]} of them are with updated params. \n'
                     f'{missing_rge_data.shape[0]} of them are with new RGEs. \n'
                     f'In {unmatched_names_fixed.shape[0]} of rows station_name was updated. \n'
                     f'In {other_rows.shape[0]} some differences were found, but were ignored determinately. \n'
                     f'List of newly added RGEs, that are not found in RIO data: {still_missing_rges} \n'
                     f'Number of rows in updated part of dict_gtprge_gen: {updated_part} \n'
                     f'Total number of rows in updated dict_gtprge_gen: {result_df.shape[0]}\n')

        return result_df

    class RegistryGenUpdater:

        """Class for checking 'dict_registry_gen' table for updates."""

        def __init__(self, rio_path, dict_path, begin_date, end_date,
                     version=0, dict_sheet_name='REGISTRY_GEN'):
            self.rio_path = rio_path
            self.dict_path = dict_path
            self.dict_sheet_name = dict_sheet_name
            self.begin_date = begin_date
            self.end_date = end_date
            self.version = version
            self.__load_data()

        def __load_data(self):
            """
            Reads dict and RIO data from the specified file paths
            """

            rio_data = pd.read_excel(self.rio_path)
            dict_data = pd.read_excel(self.dict_path, sheet_name=self.dict_sheet_name)

            # transfer column names to uppercase
            rio_data.columns = [s.upper() for s in rio_data.columns]
            dict_data.columns = [s.upper() for s in dict_data.columns]


if __name__ == "__main__":
    date_from = date(2018, 5, 1)
    date_to = date(2018, 5, 1)
    log_path = os.path.join(os.getcwd(), 'dict_checker_logs')
    logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                        level=logging.DEBUG, filename=log_path, filemode='w')
    logging.info('\nStarting...')
    rge_checker = RgeDictUpdater('data/RIO_2018_05.xlsx', 'data/hdbk_326.xlsx', date_from, date_to)
    cols = ['STATION_CODE', 'STATION_NAME', 'GTP_CODE', 'GTP_NAME', 'RGE_NAME']
    # result of the comparison
    res_comp = rge_checker.compare_gtprge_rio(['RGE_NUM'], ['RGE_CODE'], cols, draw_map=True)
    # result of the update
    res_upd = rge_checker.update_gtprge(res_comp['dict_rio'], res_comp['result_mask_dict'],
                                        res_comp['matched_rows_mask'], 'i.zemskov@skmmp.com')
    spreadsheet_path = r'D:\Work\dict_checker\data\hdbk_326.xlsx'
    to_spreadsheet(res_upd, spreadsheet_path, 'GTPRGE_GEN')
    logging.info('Finish!')
