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
import oracle_dict


def AddImage(dataset, label, database):
    path = dataset + "/" + label
    images = os.listdir(path)

    for i in range(0, len(images)):
        one = images[i]
        image = cv2.imread(path + "/" + one)
        database.adddata_kv(
            {
                "image:": image,
            },
            {
                "label": "u"+label.lower(),
                "lang": "Chinese",
            },
            {}
        )


def makeDatabase(database, image, counter=-1):
    shutil.rmtree(database, True)
    db = im_lmdb_wrapper.im_lmdb_wrapper(database)

    folders = os.listdir(image)
    folderCounter = 0

    for folder in folders:
        if folder not in oracle_dict.servant:
            continue
        folderCounter += 1
        AddImage(image, folder, db)
        counter -= 1
        if counter == 0:
            break
    db.end_this()


def get_ds(root, counter, filter=True):
    charset = {}
    db = neko_ocr_lmdb_mgmt(root, not filter, 1000)
    for i in range(len(db)):
        _, t = db.getitem_encoded_im(i)
        try:
            if t[0] == 'u' and t[1] == '6':
                if t not in charset:
                    charset[t] = 0
                charset[t] += 1
            if len(charset) == counter:
                break
        except:
            print(t)
            pass
        if i % 300 == 0:
            print(i, "of", len(db), "ds", root)
    return charset


def makept(dataset, font, counter, protodst, xdst, blacklist, servants="QWERTYUIOPASDFGHJKLZXCVBNM",
           masters="qwertyuiopasdfghjklzxcvbnm", space=None):
    if dataset is not None:
        if space is not None:
            chrset = list(set(xdst.union(get_ds(dataset, counter, False))).difference(blacklist).intersection(space))
        else:
            chrset = list(set(xdst.union(get_ds(dataset, counter, False))).difference(blacklist))
    else:
        chrset = list(set(xdst).difference(blacklist))
    engine = render_lite(os=84, fos=32)
    font_ids = [0 for c in chrset]
    meta = engine.render_core(chrset, ['[s]'], font, font_ids, False)
    meta = refactor_meta(meta, unk=len(chrset) + len(['[s]']))
    # inject a shapeless UNK.
    meta["protos"].append(None)
    meta["achars"].append("[UNK]")
    add_masters(meta, oracle_dict.servant, oracle_dict.master)
    # add_masters(meta,servants,masters);
    meta = finalize(meta)
    torch.save(meta, protodst)
    return chrset


def buildDict(db, font, counter=-1):
    path = db
    makept(path, font, counter, os.path.join(path, "dict.pt"), set(), set(), servants="", masters="")


if __name__ == '__main__':
    fontPath = "./AYJGW20200206.ttf"
    databasePath = "./packs/oracle"
    imagePath = "./sample"

    makeDatabase(databasePath, imagePath, counter=500)
    buildDict(databasePath, fontPath, counter=400)
