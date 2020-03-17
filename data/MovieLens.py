from data.Dataset import DataSet
import pdb

# Amazon review dataset
class ML20M(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/MovieLens/ml-20m/'
        self.user_record_file = 'ML20M_item_sequences.pkl'
        self.user_mapping_file = 'ML20M_user_mapping.pkl'
        self.item_mapping_file = 'ML20M_item_mapping.pkl'

        self.num_users = 129780
        self.num_items = 13663
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


class ML1M(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/MovieLens/ml-1m/'
        self.user_record_file = 'ML1M_item_sequences.pkl'
        self.user_mapping_file = 'ML1M_user_mapping.pkl'
        self.item_mapping_file = 'ML1M_item_mapping.pkl'

        self.num_users = 5950
        self.num_items = 3125
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
    data_set = Books()
    train_set, val_set, train_val_set, test_set, num_users, num_items = data_set.generate_dataset(index_shift=1)
    print(train_set[5])
    print(val_set[5])
    print(train_val_set[5])
    print(test_set[5])
    print(train_set[-3])
    print(val_set[-3])
    print(train_val_set[-3])
    print(test_set[-3])
    print(max(len(item_sequence) for item_sequence in train_set))
