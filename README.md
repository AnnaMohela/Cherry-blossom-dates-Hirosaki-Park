# Cherry blossoms in Hirosaki park
Predicting the dates of cherry blossom based on temperature with Deep Learning and GBRegression.

In this project the goal is to predict the begin, the height and the end of the cherry blossom season for Hirosaki Park in Aomori prefecture, Japan. 
The development of the blossoms is highly depending on the temperature in the weeks preceding the begin of the season. For this project folowing data was used:

https://www.kaggle.com/akioonodera/temperature-and-flower-status

The prediction is composed of two steps:
1.) predicting the temperatures until the end of May using the temperature data from the first 10 days of March with GBRegression
2.) the predicted temperatures are used as input for a neural network trained to predict the signature dates (begin, height and end of the blossoming)
 based on the daily avarage temperatures from March to May
