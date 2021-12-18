```
# K-fold cross validation fucntion
def k_fold_cross_val(k,data_set):
    if len(data_set) % k == 0:
        partitioned_set = np.reshape(data_set,(k,int(len(data_set)/k),-1))
    else:
        new_data_set = data_set.tolist()
        new_length = len(data_set) - len(data_set) % k
        left = [new_data_set[i]  for i in range(new_length, len(data_set))]
        partitioned_set = [new_data_set[i] for i in range(new_length)]
        partitioned_set = np.reshape(partitioned_set, (k, int(new_length / k), -1))
        partitioned_set = partitioned_set.tolist()
        for i in range(len(left)):
            partitioned_set[i].append(left[i])
        
    return partitioned_set
```
input:
```
a = np.array([
    [1,2,3],[4,5,6],
    [7,8,9],[10,11,12],
    [13,14,15],[16,17,18],
    [19,20,21],[22,23,24],[25,26,27],[28,29,30],
    [31,32,33],[34,35,36]
    ])
b = k_fold_cross_val(5,a)
print(b)
```
output is:
```
[[[1, 2, 3], [4, 5, 6], [31, 32, 33]], [[7, 8, 9], [10, 11, 12], [34, 35, 36]], [[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]], [[25, 26, 27], [28, 29, 30]]]
```
