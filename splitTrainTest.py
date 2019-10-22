import os
import numpy as np
import shutil

SPLIT_RATIO = 0.8

root_dir = 'splitdata'
train = '/train/'
test = '/test/'

classes = ['A', 'BA', 'BE', 'BI', 'BO', 'BU', 'CHI', 'DA', 'DE', 'DI', 'DO', 'DU', 'E', 'FU', 'GA', 'GE', 'GI', 'GO', 'GU', 'HA', 'HE', 'HI', 'HO', 'I', 'KA', 'KE', 'KI', 'KO', 'KU', 'MA', 'ME', 'MI', 'MO', 'MU', 'N', 'NA', 'NE', 'NI', 'NO', 'NU', 'O', 'PA', 'PE', 'PI', 'PO', 'PU', 'RA', 'RE', 'RI', 'RO', 'RU', 'SA', 'SE', 'SHI', 'SO', 'SU', 'TA', 'TE', 'TO', 'TSU', 'U', 'WA', 'WO', 'YA', 'YI', 'YO', 'YU', 'ZA', 'ZE', 'ZO', 'ZU']

for i in classes:
    os.makedirs(root_dir + train + i)
    os.makedirs(root_dir + test + i)

for currentCls in classes:
    src = "dataset/HiraganaTrain/" + currentCls
    
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames) * SPLIT_RATIO)])
    
    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]
    
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Testing: ', len(test_FileNames))
    print()
    
    for name in train_FileNames:
        shutil.copy(name, root_dir + train + currentCls)
        
    for name in test_FileNames:
        shutil.copy(name, root_dir + test + currentCls)