# -*- coding: utf-8 -*-
# Author      : ShiFan
# Created Date: 2024/12/1 15:23
import random
import secrets
import sys
import time
import unittest
from pathlib import Path


sys.path.append(Path(__file__).parent.parent.as_posix())


from bpt_py.bplustree import BPTree


KEY_SCOPE = 1000000
KEY_NUM = 1000000
ORDER = 100


class BPTTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.bpt = BPTree(ORDER)
        self.test_l = [0] * KEY_SCOPE
        self.test_d = dict()
        scope_list = list(range(KEY_SCOPE))
        self.keys = list()
        for key in random.sample(scope_list, KEY_NUM):
            data = secrets.token_hex()
            self.test_d[key] = data
            self.test_l[key] = data
            self.keys.append(key)
            self.bpt.insert(key, data)
        self.dict_time = [float("inf"), 0, float("-inf")]
        self.list_time = [float("inf"), 0, float("-inf")]
        self.bpt_time = [float("inf"), 0, float("-inf")]
        self.bpt_rm_time = [float("inf"), 0, float("-inf")]
        super().__init__(*args, **kwargs)

    def test_duplication(self):
        assert len(set(self.keys)) == len(self.keys)

    def test_use_time(self):
        i = 1000000
        while i:
            try:
                if not self.keys:
                    break
                tkey = random.choice(self.keys)
                st = time.time()
                data = self.test_d[tkey]
                ut = time.time() - st
                if ut < self.dict_time[0]:
                    self.dict_time[0] = ut
                self.dict_time[1] = (ut + self.dict_time[1]) / 2
                if ut > self.dict_time[2]:
                    self.dict_time[2] = ut
                st = time.time()
                data = self.test_l[tkey]
                ut = time.time() - st
                if ut < self.list_time[0]:
                    self.list_time[0] = ut
                self.list_time[1] = (ut + self.list_time[1]) / 2
                if ut > self.list_time[2]:
                    self.list_time[2] = ut
                if i % 2 == 0:
                    st = time.time()
                    data = self.bpt.get(tkey)
                    ut = time.time() - st
                    if ut < self.bpt_time[0]:
                        self.bpt_time[0] = ut
                    self.bpt_time[1] = (ut + self.bpt_time[1]) / 2
                    if ut > self.bpt_time[2]:
                        self.bpt_time[2] = ut
                else:
                    st = time.time()
                    self.bpt.remove(tkey)
                    ut = time.time() - st
                    if ut < self.bpt_rm_time[0]:
                        self.bpt_rm_time[0] = ut
                    self.bpt_rm_time[1] = (ut + self.bpt_rm_time[1]) / 2
                    if ut > self.bpt_rm_time[2]:
                        self.bpt_rm_time[2] = ut
                    self.keys.remove(tkey)
                i -= 1
            except Exception as e:
                print(self.bpt)
                raise e
        print(f"dtime: {self.dict_time}")
        print(f"ltime: {self.list_time}")
        print(f"bpttime: {self.bpt_time}")
        print(f"bptrmtime: {self.bpt_rm_time}")


if __name__ == '__main__':
    unittest.main()
