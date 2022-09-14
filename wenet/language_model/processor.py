'''
Author: zhyyao
Date: 2022-08-23 07:39:28
LastEditors: zhyyao
LastEditTime: 2022-08-23 08:01:20
Description: 

Copyright (c) 2022 by zhyyao, All Rights Reserved. 
'''
import torch
from wenet.dataset.processor import shuffle, batch
import random
import json
from torch.nn.utils.rnn import pad_sequence

def padding(data, pad_value):
    for sample in data:
        assert isinstance(sample, list)
        text_length = torch.tensor([len(x['txt_input']) for x in sample], 
                                    dtype=torch.int32)
        order = torch.argsort(text_length, descending=True)
        input_length = torch.tensor(
            [len(sample[i]['txt_input']) for i in order], dtype=torch.int32
        )
        output_length = torch.tensor(
            [len(sample[i]['label']) for i in order], dtype=torch.int32
        )
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_txt = [
            torch.tensor(sample[i]['txt_input'], dtype=torch.int64) for i in order
        ]
        sorted_syl = [
            torch.tensor(sample[i]['syl_input'], dtype=torch.int64) for i in order
        ]
        sorted_labels = [
            torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
        ]
        padding_txt = pad_sequence(sorted_txt,
                                    batch_first=True,
                                    padding_value=pad_value)
        padding_syl = pad_sequence(sorted_syl,
                                    batch_first=True,
                                    padding_value=pad_value)
        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=-1)
        
        yield {"key": sorted_keys, "txt_input": padding_txt, "syl_input":padding_syl, 
               "labels": padding_labels, "input_length": input_length, "output_length":output_length}

def filter(data,
           token_max_length=200,
           token_min_length=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'txt_input' in sample
        assert 'syl_input' in sample
    
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        if len(sample['txt_input']) < token_min_length:
            continue
        if len(sample['txt_input']) > token_max_length:
            continue
        
        yield sample

def tokenize(data,
             symbol_table,
             syl_table,
             bpe_model=None,
             non_lang_syms=None,
             split_with_space=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    if non_lang_syms is not None:
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
    else:
        non_lang_syms = {}
        non_lang_syms_pattern = None

    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    for sample in data:
        # process text 
        src = json.loads(sample['src'])
        assert 'text' in src
        txt = src['text'].strip()
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(txt.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [txt]

        label = []
        tokens = []
        for part in parts:
            if part in non_lang_syms:
                tokens.append(part)
            else:
                if bpe_model is not None:
                    tokens.extend(__tokenize_by_bpe_model(sp, part))
                else:
                    if split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = "▁"
                        tokens.append(ch)

        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])

        sample['tokens'] = tokens
        sample['txt_input'] = label

        # process syllable
        assert 'syllable' in src
        syl = src['syllable'].strip()
        parts = [syl]
        syl_tokens = []
        syl_input = []
        for part in parts:
            part = part.split(" ")
            for ch in part:
                syl_tokens.append(ch)
        for ch in syl_tokens:
            if ch in syl_table:
                syl_input.append(syl_table[ch])
            elif '<unk>' in syl_table:
                syl_input.append(syl_table['<unk>'])
        sample['syl_tokens'] = syl_tokens
        sample['syl_input'] = syl_input

        # label
        assert 'label' in src
        target = src['label'].strip()
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(target.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [target]

        label = []
        tokens = []
        for part in parts:
            if part in non_lang_syms:
                tokens.append(part)
            else:
                if bpe_model is not None:
                    tokens.extend(__tokenize_by_bpe_model(sp, part))
                else:
                    if split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = "▁"
                        tokens.append(ch)
        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])

        sample['label_tokens'] = tokens
        sample['label'] = label


        # process key
        key = src['key'].strip()
        sample['key'] = key

        #mask
        mask = src['mask'].strip()
        sample['mask'] = mask
        
        yield sample

def mask(data,
        mask_value,
        mask_prob):
    for sample in data:
        assert 'mask' in sample
        mask = sample['mask'].split()
        assert len(sample['txt_input']) == len(sample['syl_input']) == len(mask), print(
                sample['tokens'], sample['syl_tokens'], len(mask))
        for i in range(len(mask)):
            if mask[i] == 1:
                if random.random() < mask_prob:
                    sample['txt_input'][i] = mask_value
        yield sample

def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: len(x['txt_input']))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: len(x['txt_input']))
    for x in buf:
        yield x
