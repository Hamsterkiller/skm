"""
    Runner script for the updating process
    IMPORTANT: process of updating data on sheets GTPRGE_GEN, REGISTRY_GEN, HOLDINGS !must! be strict:
     GTPRGE_GEN --> REGISTRY_GEN --> HOLDINGS
     It's a little overhead of read/write excel operations, but still it's not so big.
     Maybe should refactor it later.
"""

from updaters import RgeDictUpdater, RegistryGenUpdater, HoldingsUpdater
import logging
from datetime import date
import os
from aux_functions import to_spreadsheet

if __name__ == "__main__":

    date_from = date(2018, 5, 1)
    date_to = date(2018, 5, 1)
    log_path = os.path.join(os.getcwd(), 'dict_checker_logs')
    logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                        level=logging.DEBUG, filename=log_path, filemode='w')
    rio_path = 'data/RIO_2018_05.xlsx'
    dict_path = 'data/hdbk_350.xlsx'
    dict_path_upd = 'data/hdbk_350.xlsx'
    logging.info('\nStarting...')

    # create instance of the GTPRGE_GEN updater
    rge_updater = RgeDictUpdater(rio_path, dict_path, date_from, date_to)

    # list of columns to compare by
    cols_rge = ['STATION_CODE', 'STATION_NAME', 'GTP_CODE', 'GTP_NAME', 'RGE_NAME']

    # compare grprge_gen dictionary and data in RIO file
    logging.info('Starting GTPRGE_GEN dict comparison ...')
    res_comp_rge = rge_updater.compare_gtprge_rio(['RGE_NUM'], ['RGE_CODE'], cols_rge, draw_map=True)
    logging.info('Comparison of RGE data is finished ...')

    # update RGE dictionary
    result_rge = rge_updater.update_gtprge(res_comp_rge['dict_rio'], res_comp_rge['not_found_in_rio'],
                                           res_comp_rge['result_mask_dict'], res_comp_rge['matched_rows_mask'],
                                           'i.zemskov@skmmp.com')

    # update excel file with dictionary
    logging.info('Start updating GTPRGE_GEN sheet in excel file...')
    to_spreadsheet(result_rge, dict_path_upd, 'GTPRGE_GEN')
    logging.info('Updating of GTPRGE_GEN sheet is finished.')

    # create instance of the REGISTRY_GEN updater
    registry_gen_updater = RegistryGenUpdater(rio_path, dict_path, date_from, date_to)

    # list of columns to compare by
    cols_registry = ['STATION_NAME', 'TRADER_CODE', 'TRADER_NAME']

    # compare registry_gen dictionary and data in RIO file
    logging.info('Starting REGISTRY_GEN dict comparison ...')
    res_comp_registry = registry_gen_updater.compare_registry_gen(['STATION_CODE'], ['STATION_CODE'],
                                                                  cols_registry, draw_map=True)
    logging.info('Comparison of REGISTRY_GEN data is finished ...')

    # update REGISTRY_GEN dictionary
    result_registry = registry_gen_updater.update_registry_gen(res_comp_registry['dict_rio'],
                                                               res_comp_registry['not_found_in_rio'],
                                                               res_comp_registry['result_mask_dict'],
                                                               res_comp_registry['matched_rows_mask'],
                                                               'i.zemskov@skmmp.com')

    # update excel file with dictionary
    logging.info('Start updating REGISTRY_GEN sheet in excel file...')
    to_spreadsheet(result_registry, dict_path_upd, 'REGISTRY_GEN')
    logging.info('Updating of REGISTRY_GEN sheet is finished.')

    # create instance of the REGISTRY_GEN updater
    holdings_updater = HoldingsUpdater(dict_path, 'REGISTRY_GEN', 'HOLDINGS')

    # update holdings data
    result_holdings = holdings_updater.update_holdings_info('i.zemskov@skmmp.com')

    # update excel file with dictionary

    logging.info('Start updating HOLDINGS sheet in excel file...')
    to_spreadsheet(result_holdings, dict_path_upd, 'HOLDINGS')
    logging.info('Updating of HOLDINGS sheet is finished.')
