# -*- coding:utf-8 -*-

import sys
import numpy as np
#import sentencepiece as spm
#import jieba
import re
import pickle
#from propeller import log
import itertools
#from propeller.paddle.data import Dataset

import six
import collections
import unicodedata

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


if six.PY2:
    import operator

    def accumulate(iterable, func=operator.add, initial=None):
        'Return running totals'
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
        # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        total = initial
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total = func(total, element)
            yield total
else:
    from itertools import accumulate
    import io
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    #sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

max_input_chars_per_word = 512

def wordpiece(token, vocab, unk_token, sentencepiece_style_vocab=False):
    """call with single word"""
    chars = list(token.strip())
    #chars = list(token)
    #chars = list(token.split())
    #print('chars', chars)
    if len(chars) > max_input_chars_per_word:
        return [unk_token], [(0, len(chars))]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_pos = []
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start == 0 and sentencepiece_style_vocab:
                substr = u'\u2581' + substr
            if start > 0 and not sentencepiece_style_vocab:
                if re.match("^[A-Za-z0-9]+$", substr):
                    substr = "##" + substr
                else:
                    substr = substr
                #substr = "##" + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        sub_pos.append((start, end))
        start = end
    if is_bad:
        return [unk_token], [(0, len(chars))]
    else:
        return sub_tokens, sub_pos


class JBSPTokenizer(object):
    def __init__(self, sp_model_dir, jb=True, lower=True):
        self.jb = jb
        self.lower = lower
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_dir)

    def __call__(self, sen):
        sen = sen.decode('utf8')
        if self.jb:
            sen = [s for s in jieba.cut(sen) if s != ' ']
        else:
            sen = sen.split(' ')
        if self.lower:
            sen = [s.lower() for s in sen]
        sen = ' '.join(sen)
        ret = self.sp_model.EncodeAsPieces(sen)
        return ret


class FakeJBSPTokenizer(object):
    def __init__(self, sp_model_dir, jb=True, lower=True):
        self.jb = jb
        self.lower = lower
        self.dict = pickle.load(
            open('./dict.jb_en.pickle', 'rb'), encoding='utf8')
        self.sp_model = spm.SentencePieceProcessor()
        self.window_size = 5
        self.sp_model.Load(sp_model_dir)

    def cut(self, chars):
        words = []
        idx = 0
        while idx < len(chars):
            matched = False
            for i in range(self.window_size, 0, -1):
                cand = chars[idx:idx + i]
                if cand in self.dict:
                    words.append(cand)
                    matched = True
                    break
            if not matched:
                i = 1
                words.append(chars[idx])
            idx += i
        return words

    def __call__(self, sen):
        sen = sen.decode('utf8')
        if self.jb:
            sen = [s for s in self.cut(sen) if s != ' ']
        else:
            sen = sen.split(' ')
        if self.lower:
            sen = [s.lower() for s in sen]
        sen = ' '.join(sen)
        ret = self.sp_model.EncodeAsPieces(sen)
        return ret


class SpaceTokenizer(object):
    def __init__(self, vocab, lower=True):
        """
        char tokenizer (wordpiece english)
        normed txt(space seperated or not) => list of word-piece
        """
        self.vocab = set(vocab)
        self.lower = lower

    def __call__(self, sen):
        if len(sen) == 0:
            return []  #empty line
        sen = sen.decode('utf8')
        if self.lower:
            sen = sen.lower()
        res = []
        for s in sen.split(' '):
            if s == ' ':
                continue
            if s in self.vocab:
                res.append(s)
            else:
                res.append('[UNK]')
        return res


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        #text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = convert_to_unicode(line.strip()).split("\t")
        if len(items) > 2:
            break 
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True, sentencepiece_style_vocab=False):
        """
        char tokenizer (wordpiece english)
        normed txt(space seperated or not) => list of word-piece
        """
        #self.vocab = set(vocab)
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v:k for k, v in self.vocab.items()}
        #self.pat = re.compile(r'([,.!?\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]|[\u4e00-\u9fa5]|[a-zA-Z0-9]+)')
        #self.pat = re.compile(r'([a-zA-Z0-9]+|\S)')
        self.pat = re.compile(r'([a-zA-Z0-9]+\s|\S+\s|[a-zA-Z0-9]+$|\S+$)')
        self.lower = do_lower_case
        self.sentencepiece_style_vocab = sentencepiece_style_vocab
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    def __call__(self, text):
        if len(text) == 0:
            return []  #empty line
        #sen = sen.decode('utf8')
        #sen = convert_to_unicode(sen)
        
        #if self.lower:
        #    sen = sen.lower()
        res = []
        for sen in self.basic_tokenizer.tokenize(text):
            for match in self.pat.finditer(sen):
                #print('match:', match.group(0))
                words, _ = wordpiece(
                    match.group(0),
                    vocab=self.vocab,
                    unk_token='[UNK]',
                    sentencepiece_style_vocab=self.sentencepiece_style_vocab)
                res.extend(words)
        return res

    def tokenize(self, text):
        return self.__call__(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
        #return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(i, self.inv_vocab[self.vocab['[UNK]']]) for i in ids]
        #return convert_by_vocab(self.inv_vocab, ids)



def build_2_pair(seg_a, seg_b, max_seqlen, cls_id, sep_id):
    token_type_a = np.ones_like(seg_a, dtype=np.int64) * 0
    token_type_b = np.ones_like(seg_b, dtype=np.int64) * 1
    #sen_emb = np.concatenate([[cls_id], seg_a, [sep_id], seg_b, [sep_id]], 0)
    sen_emb = np.concatenate([[cls_id], seg_a, [sep_id], seg_b], 0)
    token_type_emb = np.concatenate(
        [[0], token_type_a, [0], token_type_b, [1]], 0)

    seqlen = sen_emb.shape[0]
    #random truncate
    random_begin = 0  #np.random.randint(0, np.maximum(0, seqlen - max_seqlen) + 1,)
    #sen_emb = sen_emb[random_begin: random_begin + max_seqlen]
    sen_emb = sen_emb[random_begin:random_begin + max_seqlen - 1]
    sen_emb = np.concatenate([sen_emb, [sep_id]], 0)
    token_type_emb = token_type_emb[random_begin:random_begin + max_seqlen]

    return sen_emb, token_type_emb


def build_1_pair(seg_a, max_seqlen, cls_id, sep_id):
    token_type_a = np.ones_like(seg_a, dtype=np.int64) * 0

    #sen_emb = np.concatenate([[cls_id], seg_a, [sep_id]], 0)
    sen_emb = np.concatenate([[cls_id], seg_a], 0)
    token_type_emb = np.concatenate([[0], token_type_a, [0]], 0)

    seqlen = sen_emb.shape[0]
    #random truncate
    random_begin = 0  #np.random.randint(0, np.maximum(0, seqlen - max_seqlen) + 1,)

    #sen_emb = sen_emb[random_begin: random_begin + max_seqlen]
    sen_emb = sen_emb[random_begin:random_begin + max_seqlen - 1]
    sen_emb = np.concatenate([sen_emb, [sep_id]], 0)
    token_type_emb = token_type_emb[random_begin:random_begin + max_seqlen]
    return sen_emb, token_type_emb


def expand_dims(*args):
    func = lambda i: np.expand_dims(i, -1)
    ret = [func(i) for i in args]
    return ret


def interleave(ds1, ds2):
    def gen():
        for i, j in six.moves.zip_longest(iter(ds1), iter(ds2)):
            if i is not None:
                yield i
            if j is not None:
                yield j

    return Dataset.from_generator_func(gen)


def append_label_func(fn1, fn2=None):
    def retfn(*args):
        label = args[-1]
        args = args[:-1]
        ret1 = fn1(*args)
        if fn2 is not None:
            ret2 = fn2(label)
        else:
            ret2 = label,
        ret = tuple(ret1) + tuple(ret2)
        return ret

    return retfn


if __name__ == '__main__':

    tokenizer = FullTokenizer('./ernie3.0-5w_wordmix.v3.txt')
    #a = '我爱中华人民共和国'
    a = "这是 1975 年 11 月，袁隆平 （右三） 和同事 李必湖 （右一） 在 观察 杂交 水稻 生长 情况。"
    #a="拖动拖动"
    ans = tokenizer.tokenize(a)
    print(" ".join(ans))
    # print(' '.join(ans))
    print(tokenizer.convert_tokens_to_ids(ans))

    '''
    for i in sys.stdin:
        #print('line:', i)
        print(i)
        #i = i.encode('utf-8')
        res, _ = wordpiece(i.strip(), set(vocabs), '[UNK]')
        #print('res:',res)
        print('-'.join(res))
    '''
