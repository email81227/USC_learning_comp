from os.path import join
from pydub import AudioSegment

import numpy as np


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def get_folders(direct):
    from os import listdir
    from os.path import isdir, join

    return [folder for folder in listdir(direct) if isdir(join(direct, folder))]


def get_files(folder, format=''):
    from os import listdir
    from os.path import isfile, join
    import re

    if format == '':
        return [file for file in listdir(folder) if isfile(join(folder, file))]
    else:
        extension = re.compile(format)
        return [file for file in listdir(folder) if isfile(join(folder, file)) and extension.search(file)]


def mp3towav(fmp3, fwav, ffmpeg_path=r'D:\Develop\Python\ffmpeg\bin'):
    # Assign converter
    AudioSegment.converter = join(ffmpeg_path, 'ffmpeg.exe')

    # Import mp3 audio and then convert to export
    sound = AudioSegment.from_mp3(fmp3)
    sound.export(fwav, format="wav")


def loader(path, name, dtype='float'):
    with open(join(path, name), 'r') as f:
        data_array = np.loadtxt(f, dtype=dtype)

    return data_array


def loader_hd(path, name):
    data_array = np.load(join(path, name))

    return data_array


def saver(data_array, path, name, fmt='%.18e'):
    with open(join(path, name), 'wb') as f:
        np.savetxt(f, data_array, fmt)


def saver_hd(data_array, path, name):
    np.save(join(path, name), data_array)