"""
Created on Mai 20, 2020

@author: yhe
"""


class SimpleCharacterContainer(object):

    def __init__(self, character, label, gnt_id, folder_id):
        """
        Constructor
        @param label: int
            the label for this character object
        @param character: ndarray
            the character
        """
        self.__character = character
        self.__label = label
        self.__gnt_id = gnt_id
        self.__folder_id = folder_id

    def __getitem__(self, key):
        return

    def get_character(self):
        return self.__character

    def get_label(self):
        return self.__label

    def get_gnt_id(self):
        return self.__gnt_id

    def get_folder_id(self):
        return self.__folder_id

    def set_character(self, value):
        self.__character = value

    def set_label(self, value):
        self.__label = value

    def set_gnt_id(self, value):
        self.__gnt_id = value

    def set_folder_id(self, value):
        self.__folder_id = value

    def del_character(self):
        del self.__character

    def del_label(self):
        del self.__label

    def del_gnt_id(self):
        del self.__gnt_id

    def del_folder_id(self):
        del self.__folder_id

    def get_charater_np(self):
        return self.__character

    character = property(get_character, set_character, del_character, "character's docstring")
    label = property(get_label, set_label, del_label, "label's docstring")
    gnt_id = property(get_gnt_id, set_gnt_id, del_gnt_id, "gnt_id's docstring")
    folder_id = property(get_folder_id, set_folder_id, del_folder_id, "folder_id's docstring")


if __name__ == '__main__':
    em = SimpleCharacterContainer('c', 'l', 123, 321)
    ca = em.get_character()
    print(ca)
