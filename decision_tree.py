import math
class Floot:
    def __init__(self, value):
        self.val = value
    def __eq__(self, other):
        return (math.isclose(self.val, other.val, rel_tol=1e-8))
    def __ne__(self, other):
        return not(self.__eq__(other))
    def __lt__(self, other):
        return (self.val < other.val and self.__ne__(other))
    def __gt__(self, other):
        return (self.val > other.val and self.__ne__(other))
    def __le__(self, other):
        return (self.__lt__(other) or self.__eq__(other))
    def __ge__(self, other):
        return (self.__gt__(other) or self.__eq__(other))
    
    
training_datalist = list()
test_datalist = list()
while True:
    try:
        data = input().split()
        custom_data = {}
        for i in range(len(data)-1):
            item = data[i+1].split(':')
            custom_data[int(item[0])] = float(item[1])
            
        if data[0] != '0':
            custom_data['label'] = int(data[0])
            training_datalist.append(custom_data)
        else:
            test_datalist.append(custom_data)
            
    except EOFError:
        break
        
        
def Gini(partitions):
    num_data = sum([len(partition) for partition in partitions])
    gini_index = 0.0
    for partition in partitions:
        partition_size = len(partition)
        if partition_size == 0:
            continue
        
        class_member_count = {}
        for item in partition:
            class_member_count[item['label']] = class_member_count.get(item['label'], 0) + 1
        
        sum_impurity = 0.0
        for value in class_member_count.values():
            sum_impurity += pow((value/partition_size), 2)
        
        gini_index += (1.0 - sum_impurity) * partition_size/num_data
        
    return gini_index

# Split a dataset based on an attribute and an attribute value
def binary_split(index, value, datalist):
    left, right = list(), list()
    for data in datalist:
        if Floot(data[index]) < Floot(value):
            left.append(data)
        else:
            right.append(data)
            
    return left, right

def find_best_split(datalist, attributes):
    splitting_criteria = {'attribute':999, 'value':999, 'partitions':None}
    min_gini_index = 999
    for data in datalist:
        for attribute in attributes:
            partitions = binary_split(attribute, data[attribute], datalist)
            gini_index = Gini(partitions)
            if(Floot(gini_index) < Floot(min_gini_index)):
                splitting_criteria['attribute'] = attribute
                splitting_criteria['value'] = data[attribute]
                splitting_criteria['partitions'] = partitions
                min_gini_index = gini_index
            elif(Floot(gini_index) == Floot(min_gini_index)):
                if(attribute < splitting_criteria['attribute']):
                    splitting_criteria['attribute'] = attribute
                    splitting_criteria['value'] = data[attribute]
                    splitting_criteria['partitions'] = partitions
                    
    return splitting_criteria


def majority_voting(datalist):
    class_member_count = {}
    for item in datalist:
        class_member_count[item['label']] = class_member_count.get(item['label'], 0) + 1
        
    max_key = sorted(class_member_count.items(), key=lambda item: item[0])
    max_key.sort(key=lambda item: item[1], reverse=True)
    return max_key[0][0]
    #return max(class_member_count, key=class_member_count.get)

def grow_decision_tree(node, max_depth, depth, attributes):
    left, right = node['partitions']
    del(node['partitions'])
    
    if not left or not right:
        node['right'] = node['left'] = majority_voting(left + right)
        
    elif depth >= max_depth:
        node['left'] = majority_voting(left)
        
        node['right'] = majority_voting(right)
        
    else:
        node['left'] = find_best_split(left, attributes)
        grow_decision_tree(node['left'], max_depth, depth+1, attributes)
        
        node['right'] = find_best_split(right, attributes)
        grow_decision_tree(node['right'], max_depth, depth+1, attributes)
        
        
def predict_label(dtree_node, test_data):
    if Floot(test_data[dtree_node['attribute']]) < Floot(dtree_node['value']):
        if isinstance(dtree_node['left'], dict):
            return predict_label(dtree_node['left'], test_data)
        
        else:
            return dtree_node['left']
        
    else:
        if isinstance(dtree_node['right'], dict):
            return predict_label(dtree_node['right'], test_data)
        
        else:
            return dtree_node['right']
        
        
attributes = test_datalist[0].keys()

root = find_best_split(training_datalist, attributes)
grow_decision_tree(root, 2, 1, attributes)

for test_data in test_datalist:
    print(predict_label(root, test_data))
