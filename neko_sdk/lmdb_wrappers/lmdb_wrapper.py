import lmdb
import os
import cv2
import six
from PIL import Image


class lmdb_wrapper:
    def set_meta(self):
        self.labels = {}
        self.writers = set()

    def __init__(self, lmdb_dir):
        self.root = lmdb_dir
        os.makedirs(lmdb_dir, exist_ok=True)

        self.db = lmdb.open(lmdb_dir, map_size=int(1e12))
        self.load = 0
        self.txn = self.db.begin(write=True)

    def adddata_kv(self, ikvdict, tkvdict, rkvdict):
        iks = []
        rks = []
        tks = []
        for ik in ikvdict:
            imageKey = (ik + '-%09d' % self.load).encode()
            iks.append(imageKey)
            self.txn.put(imageKey, cv2.imencode(".png", ikvdict[ik])[1])
        for rk in rkvdict:
            rawKey = (rk + '-%09d' % self.load).encode()
            rks.append(rawKey)
            self.txn.put(rawKey, rkvdict[rk])
        for tk in tkvdict:
            tKey = (tk + '-%09d' % self.load).encode()
            tks.append(tKey)
            self.txn.put(tKey, tkvdict[tk].encode())
        if self.load % 500 == 0:
            self.txn.replace('num-samples'.encode(), str(self.load).encode())
            print("load:", self.load)
            self.txn.commit()
            del self.txn
            self.txn = self.db.begin(write=True)
        self.load += 1
        return iks, rks, tks

    def end_this(self):
        self.txn.replace('num-samples'.encode(), str(self.load).encode())
        try:
            self.txn.commit()
            self.db.close()
            self.db = None
        except:
            pass

    def __del__(self):
        if self.db is not None:
            self.end_this()
