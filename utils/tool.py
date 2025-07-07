

a = '0Un9kfUpF9Q_os_03_smpl_150_510.json'.replace('.json', '')
b = a.split('_smpl_')[0] + '.mp4'
print(a)
start_t, end_t = a.split('_')[-2], a.split('_')[-1]
print(start_t, end_t)
print(b)