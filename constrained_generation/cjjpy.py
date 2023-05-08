# -*- coding: utf-8 -*-

'''
@Author : Jiangjie Chen
@Time   : 2022/5/26 19:52
@Contact: jjchen19@fudan.edu.cn
'''

import re
import datetime
import os
import subprocess
import urllib.request, urllib.parse
import argparse
from tqdm import tqdm
import sqlite3
import requests
import socket
import logging
import io
import traceback

try:
    import ujson as json
except:
    import json

HADOOP_BIN = 'PATH=/usr/bin/:$PATH hdfs'


def LengthStats(filename, key4json=None):
    len_list = []
    thresholds = [0.8, 0.9, 0.95, 0.99, 0.999]
    with open(filename) as f:
        for line in f:
            if key4json not in ['none', None, 'None']:
                len_list.append(len(json.loads(line)[key4json].split()))
            else:
                len_list.append(len(line.strip().split()))
    stats = {
        'Max': max(len_list),
        'Min': min(len_list),
        'Avg': round(sum(len_list) / len(len_list), 4),
    }
    len_list.sort()
    for t in thresholds:
        stats[f"Top-{t}"] = len_list[int(len(len_list) * t)]

    for k in stats:
        print(f"- {k}: {stats[k]}")
    return stats


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def TraceBack(error_msg):
    exc = traceback.format_exc()
    msg = f'[Error]: {error_msg}.\n[Traceback]: {exc}'
    return msg


def Now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def TorchHLoad(filepath: str, **kwargs):
    import torch, tensorflow as tf
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, **kwargs)
    else:
        with tf.io.gfile.GFile(filepath, 'rb') as reader:
            return torch.load(io.BytesIO(reader.read()), **kwargs)


def TorchHSave(obj, filepath: str, **kwargs):
    import torch, tensorflow as tf
    if filepath.startswith("hdfs://") or remote.startswith('webhdfs://'):
        with tf.io.gfile.GFile(filepath, 'wb') as f:
            buffer = io.BytesIO()
            torch.save(obj, buffer, **kwargs)
            f.write(buffer.getvalue())
    else:
        torch.save(obj, filepath, **kwargs)


def PutHDFS(local: str, remote: str):
    import tensorflow as tf
    assert remote.startswith('hdfs://') or remote.startswith('webhdfs://')
    if not tf.io.gfile.exists(remote):
        tf.io.gfile.makedirs(remote)
    RunCmd(f'{HADOOP_BIN} dfs -put {local} {remote}')


def GetHDFS(remote: str, local: str):
    assert remote.startswith('hdfs://') or remote.startswith('webhdfs://')
    os.makedirs(local, exist_ok=True)
    RunCmd(f'{HADOOP_BIN} dfs -get {remote} {local}')


def RunCmd(command):
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res, err = pipe.communicate()
    res = res.decode('utf-8')
    err = err.decode('utf-8')
    return res, err


def AbsParentDir(file, parent='..', postfix=None):
    ppath = os.path.abspath(file)
    parent_level = parent.count('.')
    while parent_level > 0:
        ppath = os.path.dirname(ppath)
        parent_level -= 1
    if postfix is not None:
        return os.path.join(ppath, postfix)
    else:
        return ppath


def init_logger(log_file=None, log_file_level=logging.NOTSET, from_scratch=False):
    from coloredlogs import ColoredFormatter
    import tensorflow as tf

    fmt = "[%(asctime)s %(levelname)s] %(message)s"
    log_format = ColoredFormatter(fmt=fmt)
    # log_format = logging.Formatter()
    logger = logging.getLogger()
    logger.setLevel(log_file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if from_scratch and tf.io.gfile.exists(log_file):
            logger.warning('Removing previous log file: %s' % log_file)
            tf.io.gfile.remove(log_file)
        path = os.path.dirname(log_file)
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def is_qid(x, hard=False):
    if type(x) is not str: return None
    ret = re.findall('^Q\d+$', x) if hard else re.findall('Q\d+', x)
    return None if len(ret) == 0 else ret[0]


def is_pid(x, hard=False):
    if type(x) is not str: return None
    ret = re.findall('^P\d+$', x) if hard else re.findall('P\d+', x)
    return None if len(ret) == 0 else ret[0]


class MiniLutDB:
    def __init__(self, db, verbose=True):
        self.db = db
        self.conn = None
        self.verbose = verbose

    def dump_lut(self, lut_tuples, verbose=None):
        # lut_tuple: (k, v)+, iterable

        if verbose is None: 
            verbose = self.verbose
        self.conn = sqlite3.connect(self.db)
        cur = self.conn.cursor()
        cur.executescript('''
        DROP TABLE IF EXISTS lut;
        CREATE TABLE lut (
        id      TEXT PRIMARY KEY UNIQUE,
        content TEXT)''')
        self.conn.commit()

        BLOCKSIZE = 100000
        block = []
        i = 0
        iter = tqdm(lut_tuples, mininterval=0.5, disable=not verbose)
        for x in iter:
            block.append(x)
            i += 1
            if i == BLOCKSIZE:
                self.conn.executemany('INSERT OR REPLACE INTO lut (id, content) VALUES (?, ?)', block)
                block = []
                i = 0
        self.conn.executemany('INSERT OR REPLACE INTO lut (id, content) VALUES (?, ?)', block)
        self.conn.commit()

        self.close()

    def update_lut(self, lut_tuples, verbose=None):
        if verbose is None: 
            verbose = self.verbose

        self.conn = sqlite3.connect(self.db)
        BLOCKSIZE = 100000
        block = []
        i = 0
        iter = tqdm(lut_tuples, mininterval=0.5, disable=not verbose)
        for x in iter:
            block.append(x)
            i += 1
            if i == BLOCKSIZE:
                self.conn.executemany('INSERT OR REPLACE INTO lut (id, content) VALUES (?, ?)', block)
                block = []
                i = 0
        self.conn.executemany('INSERT OR REPLACE INTO lut (id, content) VALUES (?, ?)', block)
        self.conn.commit()

        self.close()

    def create_index(self):
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        # sql = ('CREATE INDEX index_lut ON lut(id);')
        # self.cur.execute(sql)
        self.cur.executescript('CREATE INDEX index_lut ON lut(id);')
        self.conn.commit()

    def get(self, x, default=None):
        if x is None: return default
        if self.conn is None:
            self.conn = sqlite3.connect(self.db)
            self.cur = self.conn.cursor()

        res = self.query_lut(self.cur, x, False)[0]
        return res if res is not None else default

    def get_chunk(self, xx):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db)
            self.cur = self.conn.cursor()
        return self.query_lut(self.conn, xx, self.verbose)

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
    
    def delete_sample(self, key, value=None):
        if self.get(key) is None: return
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        self.cur.execute('DELETE FROM lut WHERE id = ?', (key,))
        self.conn.commit()
        assert self.get(key) is None, f'delete failed: {key}'

    def query_lut(self, cur: sqlite3.Cursor, keys, verbose=True):
        values = []
        if isinstance(keys, str): keys = [keys]

        iter = tqdm(keys, mininterval=0.5, disable=not verbose)
        for k in iter:
            cur.execute('SELECT content FROM lut WHERE id = ?', (k,))
            val = cur.fetchone()
            val = val[0] if val is not None else None
            values.append(val)
        return values


def OverWriteCjjPy(root='.'):
    # import difflib
    # diff = difflib.HtmlDiff()
    cnt = 0
    golden_cjjpy = os.path.join(root, 'cjjpy.py')
    # golden_content = open(golden_cjjpy).readlines()
    for dir, folder, file in os.walk(root):
        for f in file:
            if f == 'cjjpy.py':
                cjjpy = '%s/%s' % (dir, f)
                # content = open(cjjpy).readlines()
                # d = diff.make_file(golden_content, content)
                cnt += 1
                print('[%d]: %s' % (cnt, cjjpy))
                os.system('cp %s %s' % (golden_cjjpy, cjjpy))


def ReplaceChar(file, replaced, replacer):
    print(file, replaced, replacer)
    with open(file) as f:
        data = f.readlines()
        out = open(file, 'w')
        for line in data:
            out.write(line.replace(replaced, replacer))


def DeUnicode(line):
    return line.encode('utf-8').decode('unicode_escape')


def LoadIDDict(dict_file, unify_words=False, lower=False, reverse=False):
    '''
    a\tb\n, `.dict' file
    '''
    import tensorflow as tf
    assert dict_file.endswith('.dict')
    id2label = {}
    with tf.io.gfile.GFile(dict_file, 'r') as f:
        data = f.read().split('\n')
        for i, line in enumerate(data):
            if line == '': continue
            try:
                id, label = line.split('\t')
                if reverse:
                    id, label = label, id
                _val = '_'.join(label.split()) if unify_words else label
                id2label[id] = _val.lower() if lower else _val
            except:
                pass
    return id2label


def LoadWords(file, is_file=True):
    import tensorflow as tf
    if is_file:
        with tf.io.gfile.GFile(file, 'r') as f:
            data = f.read().splitlines()
    else:
        data = file.splitlines()
    return set(map(lambda x: x.strip(), data))


def ChangeFileFormat(filename, new_fmt):
    assert type(filename) is str and type(new_fmt) is str
    spt = filename.split('.')
    if len(spt) == 0:
        return filename
    else:
        return filename.replace('.' + spt[-1], new_fmt)


def CountLines(fname):
    with open(fname, 'rb') as f:
        count = 0
        last_data = '\n'
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1:] != b'\n':
            count += 1  # Remove this if a wc-like count is needed
    return count


def SearchByKey(file, key):
    with open(file, 'r') as fin:
        while True:
            line = fin.readline()
            if not line: break
            if key in line:
                print(line, end='')


def SendEmail(subject, content, receivers=['MichaelChen0110@163.com']):
    from email.mime.text import MIMEText
    import smtplib

    # receivers got to be list.
    mail_receivers = receivers
    # mail_host = "smtp.163.com
    mail_host = "220.181.12.18"
    mail_user = "MichaelChen0110@163.com"
    mail_pass = ""
    me = socket.gethostname() + "<" + mail_user + ">"
    msg = MIMEText(content, _subtype='plain', _charset='utf-8')
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = ";".join(mail_receivers)
    try:
        server = smtplib.SMTP()
        server.connect(mail_host)
        server.login(mail_user, mail_pass)
        server.sendmail(me, mail_receivers, msg.as_string())
        server.close()
        print('Have sent the email to ' + str(mail_receivers) + '. ')
        return True
    except Exception as e:
        print(str(e))
        return False


def SortDict(_dict, reverse=True):
    assert type(_dict) is dict
    return sorted(_dict.items(), key=lambda d: d[1], reverse=reverse)


def MaxCommLen(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
    max_num = 0
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > max_num:
                    max_num = record[i + 1][j + 1]
    return max_num, ''


def lark(content='test'):
    print(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--diff', nargs=2,
                        help='show difference between two files, shown in downloads/diff.html')
    parser.add_argument('--de_unicode', action='store_true', default=False,
                        help='remove unicode characters')
    parser.add_argument('--link_entity', action='store_true', default=False,
                        help='')
    parser.add_argument('--max_comm_len', action='store_true', default=False,
                        help='')
    parser.add_argument('--search', nargs=2,
                        help='search key from file, 2 args: file name & key')
    parser.add_argument('--email', nargs=2,
                        help='sending emails, 2 args: subject & content')
    parser.add_argument('--overwrite', action='store_true', default=None,
                        help='overwrite all cjjpy under given *dir* based on *dir*/cjjpy.py')
    parser.add_argument('--replace', nargs=3,
                        help='replace char, 3 args: file name & replaced char & replacer char')
    parser.add_argument('--lark', nargs=1)
    parser.add_argument('--get_hdfs', nargs=2,
                        help='easy copy from hdfs to local fs, 2 args: remote_file/dir & local_dir')
    parser.add_argument('--put_hdfs', nargs=2,
                        help='easy put from local fs to hdfs, 2 args: local_file/dir & remote_dir')
    parser.add_argument('--length_stats', nargs=2,
                        help='simple token lengths distribution of a line-by-line file, 2 args: filename & key (or none)')

    args = parser.parse_args()

    if args.overwrite:
        print('* Overwriting cjjpy...')
        OverWriteCjjPy()

    if args.replace:
        print('* Replacing Char...')
        ReplaceChar(args.replace[0], args.replace[1], args.replace[2])

    if args.search:
        file = args.search[0]
        key = args.search[1]
        print('* Searching %s from %s...' % (key, file))
        SearchByKey(file, key)

    if args.email:
        try:
            subj = args.email[0]
            cont = args.email[1]
        except:
            subj = 'running complete'
            cont = ''
        print('* Sending email {%s, %s} to host...' % (subj, cont))
        SendEmail(subj, cont)

    if args.lark:
        try:
            content = args.lark[0]
        except:
            content = 'running complete'
        print(f'* Larking "{content}"...')
        lark(content)

    if args.get_hdfs:
        remote = args.get_hdfs[0]
        local = args.get_hdfs[1]
        print(f'* Copying {remote} to {local}...')
        GetHDFS(remote, local)

    if args.put_hdfs:
        local = args.put_hdfs[0]
        remote = args.put_hdfs[1]
        print(f'* Copying {local} to {remote}...')
        PutHDFS(local, remote)

    if args.length_stats:
        file = args.length_stats[0]
        key4json = args.length_stats[1]
        print(f'* Working on {file} lengths statistics...')
        LengthStats(file, key4json)
