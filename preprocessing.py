
import function as f

""" train_data = f.get_processed_data('train.csv')
test_data = f.get_processed_data('test.csv')

 """

train_data = f.read_file('train.csv')
test_data = f.read_file('test.csv')
print(train_data)
print(test_data)