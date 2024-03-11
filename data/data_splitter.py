import os


class DataSpliter:
    def __init__(self, data_path, domain_num=None):
        self.train_data_filenames = []
        self.train_label_list = []
        self.test_data_filenames = []
        self.test_label_list = []
        print('========加载文件缓存========')
        self.filenames_set = set(os.listdir(data_path))
        print('========文件缓存加载完毕========')
        self.domain_num = domain_num

    def get_domain_num(self, domain):
        return self.domain_num[domain]

    def clear_cache(self):
        pass

    def combine(self, index):
        pass

    def split_data(self, domain, index=None):
        pass



