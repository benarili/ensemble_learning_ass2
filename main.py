from sklearn import metrics

from data_info import DataInfo
from process import process_file
from trees import create_all_trees, create_test_train_sets

if __name__ == '__main__':
    #prepare data, preprocessing
    process_file("Adware_Multiclass_Classification.csv", 'Class',['Unnamed: 0','Flow ID'])
    process_file("Skyserver_SQL2_27_2018 6_51_39 PM.csv", 'class')
    process_file("train.csv", 'Segmentation',['ID'])
    infos = [
        (r"C:\Users\liadb\PycharmProjects\ass2-data\processed\Skyserver_SQL2_27_2018 6_51_39 PM.csv", 13,metrics.precision_score),
        (r"C:\Users\liadb\PycharmProjects\ass2-data\processed\Adware_Multiclass_Classification.csv", 83,metrics.precision_score),
        (r"C:\Users\liadb\PycharmProjects\ass2-data\processed\train.csv",9,metrics.accuracy_score)
    ]
    for path_to_csv, class_index, metric in infos:
        di = DataInfo(path_to_csv, class_index)
        #split into test and training
        X_test, X_train, y_test, y_train = create_test_train_sets(di)
        X_test=X_test.fillna(-1)
        X_train=X_train.fillna(-1)
        #create all trees with default values
        create_all_trees(metric,X_test, X_train, y_test, y_train)

        #create all tress with optimized values
        create_all_trees(metric,X_test, X_train, y_test, y_train,True)
