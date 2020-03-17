from data.Dataset import DataSet
import pdb


# Amazon review dataset
class Children(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/GoodReads/Children/'
        self.user_record_file = 'goodreads_item_sequences.pkl'
        self.user_mapping_file = 'goodreads_user_mapping.pkl'
        self.item_mapping_file = 'goodreads_item_mapping.pkl'

        self.num_users = 48296
        self.num_items = 32871
        self.vocab_size = 0

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, index_shift=1):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)

        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        user_records = self.data_index_shift(user_records, increase_by=index_shift)

        # split dataset
        train_val_set, test_set = self.split_data_sequentially(user_records, test_radio=0.2)
        train_set, val_set = self.split_data_sequentially(train_val_set, test_radio=0.1)

        return train_set, val_set, train_val_set, test_set, self.num_users, self.num_items + index_shift


class Comics(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/GoodReads/Comics/'
        self.user_record_file = 'goodreads_comics_item_sequences.pkl'
        self.user_mapping_file = 'goodreads_comics_user_mapping.pkl'
        self.item_mapping_file = 'goodreads_comics_item_mapping.pkl'

        self.num_users = 34445
        self.num_items = 33121
        self.vocab_size = 0

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, index_shift=1):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)

        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        user_records = self.data_index_shift(user_records, increase_by=index_shift)

        # split dataset
        train_val_set, test_set = self.split_data_sequentially(user_records, test_radio=0.2)
        train_set, val_set = self.split_data_sequentially(train_val_set, test_radio=0.1)

        return train_set, val_set, train_val_set, test_set, self.num_users, self.num_items + index_shift


if __name__ == '__main__':
    data_set = Children()
    train_set, val_set, train_val_set, test_set, num_users, num_items = data_set.generate_dataset(index_shift=1)
    print(train_set[0])
    print(val_set[0])
    print(train_val_set[0])
    print(train_set[-1])
    print(val_set[-1])
    print(train_val_set[-1])
    print(max(len(item_sequence) for item_sequence in train_set))
    data_set.save_pickle([train_val_set, test_set, num_users, num_items], 'Children_for_SASR', protocol=2)
