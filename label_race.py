__author__ = 'krishnateja'

def label_race(row):
    if 0 < row['last_evaluation'] <= 0.1:
        return 1
    if 0.1 < row['last_evaluation'] <= 0.2:
        return 2
    if 0.2 < row['last_evaluation'] <= 0.3:
        return 3
    if 0.3 < row['last_evaluation'] <= 0.4:
        return 4
    if 0.4 < row['last_evaluation'] <= 0.5:
        return 5
    if 0.5 < row['last_evaluation'] <= 0.6:
        return 6
    if 0.6 < row['last_evaluation'] <= 0.7:
        return 7
    if 0.7 < row['last_evaluation'] <= 0.8:
        return 8
    if 0.8 < row['last_evaluation'] <= 0.9:
        return 9
    if 0.9 < row['last_evaluation'] <= 1:
        return 1
