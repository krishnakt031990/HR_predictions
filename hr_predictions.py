import pickle
from flask import json
import requests
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

__author__ = 'krishnateja'

import pandas
import matplotlib.pyplot as plt
import numpy
from label_race import label_race
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

import seaborn

hr_data = pandas.read_csv('data/HR_comma_sep.csv')

# print('Number of people employed in the company over the period of time: ' + str(len(hr_data)))

# employees_left = hr_data.loc[hr_data['left'] == 1]
#
# print('Number of people left over this period: ' + str(len(employees_left)))

# departments = hr_data['sales']
# print(set(departments))
#
# mean_satisfaction = hr_data.groupby('sales', as_index=False)['satisfaction_level'].mean()
# print(mean_satisfaction)
#
# median_satisfaction = hr_data.groupby('sales', as_index=False)['satisfaction_level'].median()
# print(median_satisfaction)r
#
# mean_medium = pandas.merge(mean_satisfaction, median_satisfaction, how='outer', on=['sales'])
# print(mean_medium)

# mean_medium.plot(kind='bar',x=mean_medium['sales'])
# plt.show()

# hr_data['eval_person'] = hr_data.apply(lambda row: label_race(row), axis=1)
#
# good_better_best = pandas.DataFrame({'count': hr_data.groupby(["eval_person", "left"]).size()}).reset_index()
#
# count_stay = good_better_best[good_better_best.left == 0]
# count_left = good_better_best[good_better_best.left == 1]
#
# count_stay['Key'] = 'stay'
# count_left['Key'] = 'left'
#
# df_stay_leave = pandas.concat([count_stay,count_left],keys=['stay','left'])
#
# df_stay_leave_group = df_stay_leave.groupby(['eval_person','Key'])
#
# df_stay_leave_group[['count']].sum().unstack('Key').plot(kind='bar')
#
# plt.show()

# kmeans_df = data[data.left == 1].drop([u'number_project',
#                                        u'average_montly_hours', u'time_spend_company', u'Work_accident',
#                                        u'left', u'promotion_last_5years', u'sales', u'salary'], axis=1)
# kmeans = KMeans(n_clusters=3, random_state=0).fit(kmeans_df)
# print(kmeans.cluster_centers_)
#
# left = data[data.left == 1]
# left['label'] = kmeans.labels_
# plt.figure()
# plt.xlabel('Satisfaction Level')
# plt.ylabel('Last Evaluation')
# plt.plot(left.satisfaction_level[left.label==0],left.last_evaluation[left.label==0],'o', alpha = 0.2, color = 'r')
# plt.plot(left.satisfaction_level[left.label==1],left.last_evaluation[left.label==1],'x', alpha = 0.2, color = 'g')
# plt.plot(left.satisfaction_level[left.label==2],left.last_evaluation[left.label==2],'*', alpha = 0.2, color = 'b')
# plt.legend(['Performers','Frustrated','Satisfied'], loc = 3, fontsize = 15,frameon=True)
# plt.show()

# corrmat = hr_data.corr()
# seaborn.heatmap(corrmat, square=True)
# plt.show()

# print(len(test_hr_data))
# print(len(train_hr_data))

# classifiers = [('RandomForestClassifierG', RandomForestClassifier(n_jobs=-1, criterion='gini')),
# # ('RandomForestClassifierE', RandomForestClassifier(n_jobs=-1, criterion='entropy')),
# # ('AdaBoostClassifier', AdaBoostClassifier()),
# # ('ExtraTreesClassifier', ExtraTreesClassifier(n_jobs=-1)),
# # ('KNeighborsClassifier', KNeighborsClassifier(n_jobs=-1)),
# # ('DecisionTreeClassifier', DecisionTreeClassifier()),
# # ('ExtraTreeClassifier', ExtraTreeClassifier()),
# # ('LogisticRegression', LogisticRegression()),
# # ('GaussianNB', GaussianNB()),
# # ('BernoulliNB', BernoulliNB())
# ]
# allscores = []
# #
# salary_groups = {'low': 0, 'medium': 1, 'high': 2}
#
# department_groups = {'sales': 1,
#                      'marketing': 2,
#                      'product_mng': 3,
#                      'technical': 4,
#                      'IT': 5,
#                      'RandD': 6,
#                      'accounting': 7,
#                      'hr': 8,
#                      'support': 9,
#                      'management': 10
#                      }
# hr_data.salary = hr_data.salary.map(salary_groups)
# #
# hr_data['department'] = hr_data.sales.map(department_groups)
#
# # for dept in hr_data.sales.unique():
# #     hr_data['department_' + dept] = (hr_data.sales == dept).astype(int)
# hr_data = hr_data.drop('sales', axis=1)
#
# x, Y = hr_data.drop('left', axis=1), hr_data['left']
# print(x)
# for name, classifier in classifiers:
#     scores = []
#     for i in range(3):  # three runs
#         roc = cross_val_score(classifier, x, Y, scoring='roc_auc', cv=20)
#         scores.extend(list(roc))
#     classifier.fit(x, Y)
#     print(classifier)
#     filename = 'finalized_model.sav'
#     pickle.dump(classifier, open(filename, 'wb'))
#     scores = numpy.array(scores)
#     print(name, scores.mean())
#     new_data = [(name, score) for score in scores]
#     allscores.extend(new_data)
# #
# filename = 'finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# print(loaded_model)

url = "http://localhost:9000/api"
data = json.dumps(
    {'satisfaction_level': 0.38, 'last_evaluation': 0.53, 'number_project': 2, 'average_montly_hours': 157,
     'time_spend_company': 3, 'Work_accident': 0, 'promotion_last_5years': 1, 'department': 1, 'salary': 1})
r = requests.post(url, data)

print(r.json)
