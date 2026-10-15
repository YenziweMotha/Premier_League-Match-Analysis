import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = pd.read_csv("data/epl_final.csv",index_col=0)

#Making a match ID to group history of teams
df['MatchID'] = range(len(df))


#Making Matchdat into a datetime
df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
df = df.sort_values('MatchDate')

#Creating a "Fixtures" columns 
df['Fixture'] = df['HomeTeam'] +"(H) "+" vs "+ df['AwayTeam']+ " (A)"
col = df.pop('Fixture')
df.insert(1, "Fixture", col)

#Adding red cards and yellow cards of both teams
df['Total Yellow Cards'] = df[['HomeYellowCards','AwayYellowCards']].sum(axis=1)
df['Total Red Cards'] = df[['HomeRedCards','AwayRedCards']].sum(axis=1)


#Creating the points system
home_points = {'H': 3, 'D': 1, 'A': 0}
away_points = {'H': 0, 'D': 1, 'A': 3}
df['HomePoints'] = df['FullTimeResult'].map(home_points)
df['AwayPoints'] = df['FullTimeResult'].map(away_points)


#Creating Team History and team form for the past games (average points)
HomeHistory = df[['MatchID', 'MatchDate', 'HomeTeam', 'HomePoints',
                  'FullTimeHomeGoals', 'FullTimeAwayGoals']].rename(columns={
                      'HomeTeam': 'Team','HomePoints': 'Points',
                      'FullTimeHomeGoals': 'GoalsScored',
                      'FullTimeAwayGoals': 'GoalsConceded'})
                      
AwayHistory = df[['MatchID', 'MatchDate', 'AwayTeam', 'AwayPoints',
                  'FullTimeAwayGoals', 'FullTimeHomeGoals']].rename(columns={
                      'AwayTeam': 'Team',
                      'AwayPoints': 'Points',
                      'FullTimeAwayGoals': 'GoalsScored','FullTimeHomeGoals': 'GoalsConceded'})

                      
HomeHistory['Venue'] = 'Home'                    
AwayHistory['Venue'] = 'Away'
TeamHistory = pd.concat([HomeHistory, AwayHistory], ignore_index=True)
TeamHistory = TeamHistory.sort_values(['Team', 'MatchDate', 'MatchID'])

TeamHistory['FormPoints5'] = (TeamHistory.groupby('Team')['Points'].transform(
    lambda points: points.shift(1).rolling(window=5, min_periods=1).mean()))

# Average goals from the previous five matches
for column in ['GoalsScored', 'GoalsConceded']:
   TeamHistory[f'Form{column}5'] = (
        TeamHistory.groupby('Team')[column]
        .transform(
            lambda values: values.shift(1).rolling(
                window=5, min_periods=1
            ).mean()
        )
    )

# Form from previous home matches only
home_venue_history = TeamHistory[TeamHistory['Venue'] == 'Home'].copy()
home_venue_history = home_venue_history.sort_values(['Team', 'MatchDate', 'MatchID'])
home_venue_history['HomeVenueForm5'] = (home_venue_history.groupby('Team')['Points'].transform(
    lambda points: points.shift(1).rolling(
            window=5, min_periods=1
        ).mean()
    )
)

# Form from previous away matches only
away_venue_history = TeamHistory[TeamHistory['Venue'] == 'Away'].copy()
away_venue_history = away_venue_history.sort_values(['Team', 'MatchDate', 'MatchID'])
away_venue_history['AwayVenueForm5'] = (away_venue_history.groupby('Team')['Points'].transform(
        lambda points: points.shift(1).rolling(
            window=5, min_periods=1
        ).mean()
    )
)

home_form = TeamHistory[['MatchID', 'Team', 'FormPoints5']].rename(columns={'Team': 'HomeTeam',
                                                                            'FormPoints5': 'HomeForm5'})
away_form =TeamHistory[['MatchID', 'Team', 'FormPoints5']].rename(columns={'Team': 'AwayTeam',
                                                                           'FormPoints5': 'AwayForm5'})
    
print(TeamHistory[['MatchDate', 'Team', 'GoalsScored', 'GoalsConceded']].head(10))
print(TeamHistory[['MatchDate', 'Team', 'GoalsScored', 'GoalsConceded','FormGoalsScored5', 'FormGoalsConceded5']].head(10))
print("Missing scoring form:",TeamHistory['FormGoalsScored5'].isna().sum())
print("Missing conceding form:",TeamHistory['FormGoalsConceded5'].isna().sum())
print(TeamHistory.groupby('Venue').size())
print("Team history rows:", len(TeamHistory))
print("Missing scored goals:", TeamHistory['GoalsScored'].isna().sum())
print("Missing conceded goals:", TeamHistory['GoalsConceded'].isna().sum())
print(TeamHistory[['MatchDate', 'Team', 'Points', 'FormPoints5']].head(10))
print("Missing form values:", TeamHistory['FormPoints5'].isna().sum())
print("Duplicate team/date records:",TeamHistory.duplicated(subset=['Team', 'MatchDate']).sum())
print("Missing home venue form:",home_venue_history['HomeVenueForm5'].isna().sum())
print("Missing away venue form:",away_venue_history['AwayVenueForm5'].isna().sum())

# Look up each team's historical form for its match
HomeForm = TeamHistory[['MatchID', 'Team', 'FormPoints5']].rename(columns={'Team': 'HomeTeam',
                                                                           'FormPoints5': 'HomeForm5'})
AwayForm = TeamHistory[['MatchID', 'Team', 'FormPoints5']].rename(columns={'Team': 'AwayTeam','FormPoints5': 'AwayForm5'})
df = df.merge(HomeForm,on=['MatchID', 'HomeTeam'],how='left')
df = df.merge(AwayForm ,on=['MatchID', 'AwayTeam'],how='left')

# Prepare home team's historical goal form
home_goal_form = TeamHistory[['MatchID', 'Team', 'FormGoalsScored5', 
                              'FormGoalsConceded5']].rename(columns={'Team': 'HomeTeam',
'FormGoalsScored5': 'HomeGoalsScored5','FormGoalsConceded5': 'HomeGoalsConceded5'})

# Prepare away team's historical goal form
away_goal_form = TeamHistory[['MatchID', 'Team', 
                              'FormGoalsScored5', 'FormGoalsConceded5']].rename(columns={'Team': 'AwayTeam','FormGoalsScored5': 'AwayGoalsScored5','FormGoalsConceded5': 'AwayGoalsConceded5'})

# Attach historical goal form to each fixture
df = df.merge(home_goal_form,on=['MatchID', 'HomeTeam'],how='left')
df = df.merge(away_goal_form,on=['MatchID', 'AwayTeam'],how='left')

home_venue_form = home_venue_history[['MatchID', 'Team', 'HomeVenueForm5']].rename(columns={'Team': 'HomeTeam'})
away_venue_form = away_venue_history[['MatchID', 'Team', 'AwayVenueForm5']].rename(columns={'Team': 'AwayTeam'})
df = df.merge(home_venue_form,on=['MatchID', 'HomeTeam'], how='left')
df = df.merge(away_venue_form, on=['MatchID', 'AwayTeam'],how='left')

print("Rows after venue merges:", len(df))
print(df[['HomeVenueForm5', 'AwayVenueForm5']].isna().sum())
print(df[['MatchDate', 'HomeTeam', 'AwayTeam','HomeGoalsScored5', 'HomeGoalsConceded5','AwayGoalsScored5', 'AwayGoalsConceded5']].head(10))
print("Total match rows:", len(df))
print("Missing home goals form:",df['HomeGoalsScored5'].isna().sum())
print("Missing away goals form:",df['AwayGoalsScored5'].isna().sum())
print(df[['MatchDate', 'HomeTeam', 'AwayTeam','HomeForm5', 'AwayForm5']].head(10))
print("Missing home form:", df['HomeForm5'].isna().sum())
print("Missing away form:", df['AwayForm5'].isna().sum())
print("Match rows after merging:", len(df))


#adding corners of both teams
df['Total Corners'] = df[['HomeCorners','AwayCorners']].sum(axis=1)

#Creating a numerical predictor for Full time Result column
predictor_map =  {'H':1, 'D':0, 'A':2}
df['Result_Value'] = df['FullTimeResult'].map(predictor_map)
df['Result_Value'] = pd.to_numeric(df['Result_Value'], errors="coerce").astype('Int64')


#Training Model
categorical_features = ['HomeTeam', 'AwayTeam']
numeric_features = ['HomeForm5','AwayForm5','HomeGoalsScored5',
                    'HomeGoalsConceded5','AwayGoalsScored5','AwayGoalsConceded5',
                    'HomeVenueForm5','AwayVenueForm5']

predictors = categorical_features + numeric_features

# Prepare team names and historical form
preprocessor = ColumnTransformer(transformers=[('teams', OneHotEncoder(handle_unknown='ignore'),
                                                categorical_features),
                                               ('form', SimpleImputer(strategy='median'),numeric_features)])

# Build the model pipeline
rf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', RandomForestClassifier(n_estimators=90,min_samples_split=5,
                                                           random_state=1, class_weight='balanced'))])

# Split chronologically
train = df[df['MatchDate'] < pd.Timestamp('2020-01-01')]
test = df[df['MatchDate'] >= pd.Timestamp('2020-01-01')]

# Train and predict
rf.fit(train[predictors], train['Result_Value'])
preds = rf.predict(test[predictors])


acc = accuracy_score(test['Result_Value'], preds)
print(acc)
combined = pd.DataFrame(dict(actual = test["Result_Value"], prediction=preds))
combined= pd.crosstab(index = combined["actual"], columns=combined["prediction"])
print(combined)

#precision score
from sklearn.metrics import precision_score
pres_score = precision_score(test['Result_Value'], preds, average='macro')
print(pres_score)

# Compare against a simple baseline
majority_class = train['Result_Value'].mode().iloc[0]
baseline_preds = [majority_class] * len(test)

baseline_accuracy = accuracy_score(
    test['Result_Value'], baseline_preds)

print("Training majority class:", majority_class)
print("Baseline accuracy:", baseline_accuracy)

print("\nClassification report:")
print(classification_report(test['Result_Value'],preds,labels=[0, 1, 2],
                            target_names=['Draw', 'Home win', 'Away win'],zero_division=0))
print(TeamHistory.head(10))
print("History rows:", len(TeamHistory))
print("Unique teams:", TeamHistory['Team'].nunique())
print("Missing points:", TeamHistory['Points'].isna().sum())