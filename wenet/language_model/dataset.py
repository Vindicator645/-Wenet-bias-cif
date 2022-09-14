'''
Author: zhyyao
Date: 2022-08-23 07:31:58
LastEditors: zhyyao
LastEditTime: 2022-08-23 07:32:34
Description: 

Copyright (c) 2022 by zhyyao, All Rights Reserved. 
'''


from wenet.dataset.dataset import *
from wenet.utils.file_utils import read_lists
import wenet.language_model.processor as processor

def Dataset_s2c(
        data_type,
        data_list_file,
        symbol_table,
        syl_table,
        conf,
        bpe_model=None,
        non_lang_syms=None,
        partition=True
    ):
    assert data_type in ['raw']
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    dataset = Processor(dataset, processor.tokenize, symbol_table, syl_table, bpe_model,
                        non_lang_syms, conf.get('split_with_space', False))
    dataset = Processor(dataset, processor.mask, symbol_table['<mask>'], conf.get('mask_prob', 1.0))
    
    filter_conf = conf.get('filter_conf', {})
    print(filter_conf)
    dataset = Processor(dataset, processor.filter, **filter_conf)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)
    
    sort = conf.get('sort',True)
    if sort:
        sort_conf = conf.get('sort_conf',{})
        dataset = Processor(dataset, processor.sort, **sort_conf)
    
    batch_conf = conf.get('batch_conf',{})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding, 0)
    return dataset
