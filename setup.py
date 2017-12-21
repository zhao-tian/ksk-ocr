import os
import shutil
import random

DATA_ROOT = "/Volumes/deepstation/ocr-20171221/"
IN_DIR = DATA_ROOT + 'jpg'
TRAIN_DIR = DATA_ROOT + 'train_images'
TEST_DIR = DATA_ROOT + 'test_images'

if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

# name => (start idx, end idx)
flower_dics = {}

with open('labels.txt') as fp:
    for line in fp:
        line = line.rstrip()
        cols = line.split()

        assert len(cols) == 3

        start = int(cols[0])
        end = int(cols[1])
        name = cols[2]

        flower_dics[name] = (start, end)

# 花ごとのディレクトリを作成
for name in flower_dics:
    ClsTrainPath = os.path.join(TRAIN_DIR, name)
    if not os.path.exists(ClsTrainPath):
        os.mkdir(ClsTrainPath)
    ClsTestPath = os.path.join(TEST_DIR, name)
    if not os.path.exists(ClsTestPath):
        os.mkdir(ClsTestPath)
# jpgをスキャン
for f in sorted(os.listdir(IN_DIR)):
    # image_0001.jpg => 1
    prefix = f.replace('.jpg', '')
    print(prefix)
    idx = int(prefix.split('_')[1])

    for name in flower_dics:
        start, end = flower_dics[name]
        if idx in range(start, end + 1):
            source = os.path.join(IN_DIR, f)
            dest = os.path.join(TRAIN_DIR, name)
            shutil.copy(source, dest)
            continue

# 訓練データの各ディレクトリからランダムに10枚をテストとする
for d in os.listdir(TRAIN_DIR):
    files = os.listdir(os.path.join(TRAIN_DIR, d))
    random.shuffle(files)
    for f in files[:10]:
        source = os.path.join(TRAIN_DIR, d, f)
        dest = os.path.join(TEST_DIR, d)
        shutil.move(source, dest)
