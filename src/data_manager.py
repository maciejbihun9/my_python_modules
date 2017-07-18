

class DataManager:
    """
    This class was designed to parse each .txt, .csv files into data list for data processing
    """
    @staticmethod
    def load_data(url: str, miss_first_line: bool) -> list:
        """
        :param url: file to parse url
        :param miss_first_line: miss first line bool value
        :return: parsed data as data_list
        """
        try:
            file = open(url)
            if miss_first_line:
                next(file)
            data = []
            for line in file.readlines():
                line_elements = line.strip().split(" ")
                data_item = []
                for line_el_index in range(len(line_elements)):
                    item = line_elements[line_el_index]
                    try:
                        item = float(line_elements[line_el_index])
                    except:
                        pass
                    data_item.extend([item])
                data.append(data_item)
        finally:
            file.close()
        return data