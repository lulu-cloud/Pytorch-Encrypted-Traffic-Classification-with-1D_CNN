import gzip

train_data_path = "data/12class/SessionAllLayers/t10k-images-idx3-ubyte.gz"
train_label_path = "data/12class/SessionAllLayers/t10k-labels-idx3-ubyte.gz"

f = gzip.open(train_data_path,'rb')

for line in f.readlines():
    s = line.decode()
    print(s)

