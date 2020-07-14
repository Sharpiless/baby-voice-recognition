LABELS = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable'] # 标签

N_CLASS = len(LABELS) # 类别数

TIME_SEG = 5 # 5s间隔

STRIDE = 3 # 1/3的滑窗长度步长

SR = 22050 # 采样频率

N_MEL = 128 # 梅尔数

EPOCHES = 400 # epoch数目

BATCH_SIZE = 256 # batch大小
