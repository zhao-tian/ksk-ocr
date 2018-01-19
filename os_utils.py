import os
import glob
import random
import shutil

def createDir (dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)

def getSubDirListOf(dirPath):
    dirList = []
    for x in os.listdir(dirPath):
        xPath = dirPath + "/" + x
        """
        if not os.path.isabs(xPath):
            xPath = os.path.dirname(os.path.abspath(__file__)) + "/" + xPath
        """
        if os.path.isdir(xPath):
            dirList.append(x)
        """
        else:
            print(x + " is not dir")
        """
    print(dirPath)
    print(len(dirList))
    return dirList

def getFileListOf(dirPath, ext):
    fileList = []
    for x in os.listdir(dirPath):
        if x.endswith('.' + ext) and not x.startswith('.'):
            fileList.append(x)
    return fileList

def getPathListOf(dirPath, ext):
    fileList = []
    for x in os.listdir(dirPath):
        if x.endswith('.' + ext) and not x.startswith('.'):
            fileList.append(dirPath + "/" + x)
    return fileList

def countFilesOf(rootDirPath, ext):
    dirList = getSubDirListOf(rootDirPath)
    count = 0
    for i in range(len(dirList)):
        if not dirList[i].startswith('.'):
            dirPath = rootDirPath + "/" + dirList[i]
            # get files
            fileList = getFileListOf(dirPath, ext)
            count += len(fileList)
            print(os.path.basename(dirPath) + " :" + str(len(fileList)))
    return count