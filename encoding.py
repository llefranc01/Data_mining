import pandas as pd

data = pd.read_excel('Customer_Churn.xlsx')

leave = {'STAY': 0, 'LEAVE': 1}
college = {'zero': 0, 'one': 1}

reported_satisfaction = {'very_unsat': 0, 'unsat': 1, 'avg': 2, 'sat': 3, 'very_sat': 4}
reported_usage_level = {'very_little': 0, 'little': 1, 'avg': 2, 'high': 3, 'very_high': 4}
considering_change_of_plan = {'never_thought': 0, 'no': 1, 'perhaps': 2, 'considering': 3,
                              'actively_looking_into_it': 4}

for i, (v1, v2, v3, v4, v5) in enumerate(
        zip(data['REPORTED_SATISFACTION'], data['REPORTED_USAGE_LEVEL'], data['CONSIDERING_CHANGE_OF_PLAN'],
            data['LEAVE'], data['COLLEGE'])):
    data.loc[i, 'REPORTED_SATISFACTION'] = reported_satisfaction[v1]
    data.loc[i, 'REPORTED_USAGE_LEVEL'] = reported_usage_level[v2]
    data.loc[i, 'CONSIDERING_CHANGE_OF_PLAN'] = considering_change_of_plan[v3]
    data.loc[i, 'LEAVE'] = leave[v4]
    data.loc[i, 'COLLEGE'] = college[v5]
data.to_csv('processed.csv', index=False)