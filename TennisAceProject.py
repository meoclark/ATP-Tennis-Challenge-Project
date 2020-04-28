#Project Goals
#To create a linear regression model that predicts the outcome for a tennis player based on their playing habits.
#By analyzing and modeling the Association of Tennis Professionals (ATP) data, 
#To determine what it takes to be one of the best tennis players in the world.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# load and investigate the data here:
tennis_data = pd.read_csv("tennis_stats.csv")
print(tennis_data.head())
print(tennis_data["Player"])
print(tennis_data.columns)
print(tennis_data.describe())



# perform exploratory analysis here:
print(tennis_data.corr())
y = tennis_data["Winnings"]
x = tennis_data["BreakPointsOpportunities"]
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')
plt.scatter(x,y, marker = '*', alpha = 0.5)
plt.show()
plt.clf()

plt.title('TotalServicePointsWon vs Wins')
plt.scatter(tennis_data['TotalServicePointsWon'],tennis_data['Wins'],marker = '*', alpha = 0.5,color = 'orange')
plt.xlabel('TotalServicePointsWon')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(tennis_data['TotalServicePointsWon'],tennis_data['Wins'],marker = '*', alpha = 0.5,color = 'red')
plt.title('TotalServicePointsWon vs Wins')
plt.xlabel('TotalServicePointsWon')
plt.ylabel('Wins')
plt.show()
plt.clf()

## I classified my model as singleM which means Single Model. others can name it model or what they prefer.

#Single feature linear regressions with BreakPointsOpportunities :

x_features = tennis_data[["BreakPointsOpportunities"]]
y_winnings = tennis_data[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
singleM = LinearRegression()
singleM.fit(x_train, y_train)
pred = singleM.predict(x_test)
print('Prediction of Winnings with BreakPointsOpportunities Test Score:',singleM.score(x_test,y_test))
plt.scatter(y_test,pred,marker = 'o', alpha = 0.4)
plt.title('Predicted Winnings vs. Actual Winnings with 1 Feature')
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()



#Another Single Feature Linear Regression with ServiceGamesPlayed
x_features = tennis_data[["ServiceGamesPlayed"]]
y_wins = tennis_data[["Wins"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_wins, train_size = 0.8, test_size = 0.2)
singleM = LinearRegression()
singleM.fit(x_train, y_train)
pred1 = singleM.predict(x_test)
print('Prediction of Wins with ServiceGamesPlayed Test Score:',singleM.score(x_test,y_test))
plt.scatter(y_test,pred1,marker = 'o', alpha = 0.4)
plt.title('Predicted Winnings vs. Actual Winnings with 1 Feature')
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()


#Another Single Feature Linear Regression with BreakPointsOpportunities
x_features = tennis_data[["BreakPointsOpportunities"]]
y_losses = tennis_data[["Losses"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_losses, train_size = 0.8, test_size = 0.2)
singleM = LinearRegression()
singleM.fit(x_train, y_train)
pred2 = singleM.predict(x_test)
print('Prediction of Losses with "BreakPointsOpportunities" Test Score:',singleM.score(x_test,y_test))
plt.scatter(y_test,pred2,marker = 'o', alpha = 0.4)
plt.title('Predicted Winnings vs. Actual Winnings with 1 Feature')
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()


#I classified my model as double which represent two features

# Two features linear regression
x_features = tennis_data[["BreakPointsOpportunities",'FirstServeReturnPointsWon']]
y_winnings = tennis_data[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
double = LinearRegression()
double.fit(x_train, y_train)
pred = double.predict(x_test)
print('Prediction of Winnings with 2 Features Test Score:',double.score(x_test,y_test))
plt.scatter(y_test,pred, alpha = 0.4)
plt.title("Predicted outcome vs Actual outcome with two features")
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()

#Another two features linear regression
x_features = tennis_data[["BreakPointsOpportunities",'ServiceGamesPlayed']]
y_winnings = tennis_data[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
double = LinearRegression()
double.fit(x_train, y_train)
pred = double.predict(x_test)
print('Prediction of Winnings with 2 Features Test Score:', double.score(x_test,y_test))
plt.scatter(y_test,pred, alpha = 0.4)
plt.title("Predicted outcome vs Actual outcome with two features")
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()


#I classified my model as multiple which represent multiple features

## Multiple features linear regressions
x_features = tennis_data[["BreakPointsOpportunities",'ServiceGamesPlayed',"TotalPointsWon","TotalServicePointsWon","DoubleFaults","BreakPointsConverted",'SecondServeReturnPointsWon']]
y_winnings = tennis_data[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
multiple = LinearRegression()
multiple.fit(x_train, y_train)
pred = multiple.predict(x_test)
print('Predicting Winnings with Multiple Features Test Score:',multiple.score(x_test,y_test))
plt.scatter(y_test,pred, alpha = 0.4)
plt.title("Predicted outcome vs Actual outcome with Multiple features")
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()



#Another Multiple
x_features = tennis_data[["BreakPointsOpportunities",'ServiceGamesPlayed',"TotalPointsWon","TotalServicePointsWon","BreakPointsConverted",'SecondServeReturnPointsWon','BreakPointsConverted',"BreakPointsFaced","ReturnGamesWon","SecondServePointsWon"]]
y_winnings = tennis_data[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
multiple = LinearRegression()
multiple.fit(x_train, y_train)
pred = multiple.predict(x_test)
print('Predicting Winnings with Multiple Features Test Score:', multiple.score(x_test,y_test))
plt.title("Predicted outcome vs Actual outcome with Multiple features")
plt.scatter(y_test,pred, alpha = 0.4)
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()














# I am happy to complete this project. Moses Egbo (Meoclark)