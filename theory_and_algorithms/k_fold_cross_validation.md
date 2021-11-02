"""
# K-fold cross validation fucntion
def k_fold_cross_val(k,data_set):
    if len(data_set) % k == 0:
        partitioned_set = np.reshape(data_set,(k,int(len(data_set)/k),-1))
    else:
        new_data_set = data_set.tolist()
        new_length = len(data_set) - len(data_set) % k
        left = [new_data_set[i]  for i in range(new_length, len(data_set))]
        partitioned_set = [element for element in new_data_set if element not in left]
        partitioned_set = np.reshape(partitioned_set, (k, int(new_length / k), -1))
        partitioned_set = partitioned_set.tolist()
        for i in range(len(left)):
            partitioned_set[i].append(left[i])
        
    return partitioned_set
"""
