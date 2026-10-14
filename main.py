import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

df = pd.read_csv("data/epl_final.csv",index_col=0) 

#print(df.info)
#print(df.head())
#print(df.shape)


#Creating a "Fixtures" columns and dropping "Away team and Home Team" columns
df['Fixture'] = df['HomeTeam'] +"(H) "+" vs "+ df['AwayTeam']+ " (A)"
col = df.pop('Fixture')
df.insert(1, "Fixture", col)
df = df.drop(columns=['HomeTeam','AwayTeam'])

#Adding red cards and yellow cards of both teams
df['Total Yellow Cards'] = df[['HomeYellowCards','AwayYellowCards']].sum(axis=1)
df['Total Red Cards'] = df[['HomeRedCards','AwayRedCards']].sum(axis=1)

#Making Matchdat into a datetime
df['MatchDate'] = pd.to_datetime(df['MatchDate'])

#adding corners of both teams
df['Total Corners'] = df[['HomeCorners','AwayCorners']].sum(axis=1)

#Creating a numerical predictor for Full time Result column
predictor_map =  {'H':1, 'D':0, 'A':2}
df['Result_Value'] = df['FullTimeResult'].map(predictor_map)
df['Result_Value'] = pd.to_numeric(df['Result_Value'], errors="coerce").astype('Int64')

#Creating the EDA

#Dropping the non neccassary columns
df = df.drop(columns=['HomeShots','AwayShots','HomeShotsOnTarget','AwayShotsOnTarget',
                      'AwayFouls','HomeFouls','HomeYellowCards','HomeRedCards',
                      'AwayYellowCards','AwayRedCards','FullTimeHomeGoals',
                      'FullTimeAwayGoals','HalfTimeResult'])

#Training Model
rf = RandomForestClassifier(n_estimators=90, min_samples_split=5, random_state=1)

train = df[df['MatchDate'] < '2020-01-01']
test = df[df['MatchDate'] >'2020-01-01']
predictors = ['Total Yellow Cards', 'Total Red Cards', 'Total Corners']
rf.fit(train[predictors], train['Result_Value'])
preds= rf.predict(test[predictors])

from sklearn.metrics import accuracy_score
acc = accuracy_score(test['Result_Value'], preds)
print(acc)
combined = pd.DataFrame(dict(actual = test["Result_Value"], prediction=preds))
combined= pd.crosstab(index = combined["actual"], columns=combined["prediction"])
print(combined)

#precision score
from sklearn.metrics import precision_score
pres_score = precision_score(test['Result_Value'], preds, average='macro')
print(pres_score)