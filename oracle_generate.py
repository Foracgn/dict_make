from neko_sdk.lmdb_wrappers import im_lmdb_wrapper, lmdb_wrapper
import shutil
import cv2
import os
import glob
import torch
import regex

from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt
from neko_sdk.ocr_modules.renderlite.addfffh import refactor_meta, add_masters, finalize
from neko_sdk.ocr_modules.renderlite.lib_render import render_lite


def AddImage(path, database):
    with open(path, "r") as f:
        for one in f:
            image = cv2.imread(one)
            database.adddata_kv(
                {
                    "image:": image,
                },
                {
                    "label": one,
                    "lang": "Chinese",
                    "attr": ""
                },
                {}
            )


def makeDatabase(database, image):
    shutil.rmtree(database, True)
    db = im_lmdb_wrapper.im_lmdb_wrapper(database)

    with open(image, "r") as fp:
        for subFolder in fp:
            AddImage(subFolder, db)
    db.end_this()


def get_ds(root, filter=True):
    charset = {}
    db = neko_ocr_lmdb_mgmt(root, not filter, 1000)
    for i in range(len(db)):
        _, t = db.getitem_encoded_im(i)
        try:
            for c in regex.findall(r'\X', t, regex.U):
                if c not in charset:
                    charset[c] = 0
                charset[c] += 1
        except:
            print(t)
            pass
        if i % 300 == 0:
            print(i, "of", len(db), "ds", root)
    return charset


def makept(dataset, font, protodst, xdst, blacklist, servants="QWERTYUIOPASDFGHJKLZXCVBNM",
           masters="qwertyuiopasdfghjklzxcvbnm", space=None):
    if dataset is not None:
        if space is not None:
            chrset = list(set(xdst.union(get_ds(dataset, False))).difference(blacklist).intersection(space))
        else:
            chrset = list(set(xdst.union(get_ds(dataset, False))).difference(blacklist))
    else:
        chrset = list(set(xdst).difference(blacklist))
    engine = render_lite(os=84, fos=32)
    font_ids = [0 for c in chrset]
    meta = engine.render_core(chrset, ['[s]'], font, font_ids, False)
    meta = refactor_meta(meta, unk=len(chrset) + len(['[s]']))
    # inject a shapeless UNK.
    meta["protos"].append(None)
    meta["achars"].append("[UNK]")
    if len(masters):
        add_masters(meta, servants, masters)
    # add_masters(meta,servants,masters);
    meta = finalize(meta)
    torch.save(meta, protodst)
    return chrset


def buildDict(db, font):
    path = os.path.join(db, "*.mdb")
    makept(path, font, os.path.join(path, "dict.pt"), set(), set(), servants="", masters="")


if __name__ == '__main__':
    fontPath = "./AYJGW20200206.ttf"
    databasePath = "./packs/oracle"
    imagePath = "./sample"

    makeDatabase(databasePath, imagePath)
    buildDict(databasePath, fontPath)
