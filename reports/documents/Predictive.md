# Predictive Task - Predict Drivers that come in second between 2011 and 2017 from 2011 to 2017 using data from 1950 to 2010

>(2) [25pts] Now we move on to prediction. Fit a model using data from 1950:2010, and predict drivers that come in second place between 2011 and 2017. [Remember, this is a predictive model where variables are selected as the subset that is best at predicting the target variable and not for theoretical reasons. This means that your model should not overfit and most likely be different from the model in (1).]
>From your fitted model:
- describe your model, and explain how you selected the features that were selected
- provide statistics that show how good your model is at predicting, and how well it performed predicting second places in races between 2011 and 2017
- the most important variable in (1) is bound to also be included in your predictive model. Provide marginal effects or some metric of importance for this variable and make an explicit comparison of this value with the values that you obtained in (1). How different are they? Why are they different?

## Introduction
In our previous inferential task (see [`Inferential.md`](reports/documents/Inferential.md)), we sought to explain reasons why drivers came in second place. While understanding the sport at a conceptual level has its appeal, being able to predict positions is also of interest to competitors and fans. Predictive models can help constructors determine whether a driver is worth investing in by tracking their predicted positions in races with the constructor, or can help plan strategy. In fact, teams like Mercedes, which has dominated the championship in recent years, [are already considering employing predictive analytics in their races](https://www.datanami.com/2018/04/19/go-fast-and-win-the-big-data-analytics-of-f1-racing/). For fans, predictive analytics may bring an additional component to fan discussion and speculation.

Here, we build on the inferential task from before to try and build a predictive model that can predict drivers that come in second between 2011 and 2017, using data from 1950 to 2010.

## Features

It is important for features to be time invariant - otherwise, the values of features the model is trained on will be entirely different from the values of the features the model will be tested on. Hence, otherwise potentially useful features like drivers, circuits, and constructors have to be excluded. A model that we train on old data believing that Cooper cars on Brands Hatch guarantees second place would find it difficult to deal with the 2010s, where there are no Cooper cars, and no Brands Hatch race.

### Features significant in inferential model

#### Starting Grid Position

As noted in the inferential task writeup, a higher starting grid position is likely to advantage a higher placed finish due to overtaking being more difficult than staying in the lead.

Startin grid position was found to be the most important variable in explaining why a driver came in second in the inferential task. It was important both in explaining finishing in the top 2 positions, and also in distinguishing first from second.

Thus, Starting Grid Position is likely to be useful for predicting 2nd place finishes.

#### Pit Strategy

Different strategies create different race dynamics, juggling the need to overtake with faster speeds of the car.

Pit strategy was also found to explain top 2 finishes, and in the case of a single pit strategy, was significant in differentiating first from second. Pit strategy also had some significant conditional relationships with grid position.

Pit strategy may be useful for predicting 2nd place finishes, given findings from the inferential model. However, because most non-missing pit data comes from later races, there may not be sufficient data in the training set to match against the more recent races of the test set. If this is the case, pit strategy and any other pit-stop based feature cannot be used.

#### Constructor quality

A good constructor brings a stronger pit team, better car, and better training to a race. Constructor quality was found to be a significant predictor in the inferential model for distinguishing top 2 finishes from the rest of the competitors. The quality of the constructor, operationalised here in the same way, should therefore also be predictive of second place finishes.

### Other features

#### Driver Experience

Driver experience was not siginificant in either inferential model for top 2 finishes or distinguishing first from second. However, in the former case, it approached significance, suggesting that it may have some predictive power in determining second place positioning.

#### Driver Average Finishing Position
In discussing the lack of significance of driver experience in the inferential model, I mentioned that driver experience might not fully capture the construct of driver skill. Some drivers may be very talented and win races from the start. Driver average finishing position would capture some of this aspect of driver skill and likely hold some predictive power.

#### Constructor Average Finishing Position
In the same way, constructor average finishing position may be a better metric of constructor quality, which is already considered to be a potential predictor above.

#### Average Qualifying lap time
Qualifying lap time did not make sense to include in the inferential model as it held little theoretical value. Qualifiying lap time as a significant model term (after controlling for the grid position earned) will simply tell you "fast tends to win", which I think is safe to assume would not even satisfy a 5 year old. However, such a direct proxy of "fast car" will likely be valuable in predicting future race outcomes in the time between the Saturday and Sunday races.

The average is used because
* Eliminations mean that Q2 and Q1 data are missing for many drivers in each race
* Gets around rule changes in the past where only a single qualifying lap was used, which creates "missing" Q2 and Q1 data that cannot be simply imputed by using the max value (as you could reasonably do for missing actual Q2/Q1 data)
* For drivers where Q2 and Q1 data are present, an average provides a more robust estimate of their speed

#### Q3 Qualifying Lap Time
While the average provides more robust estimates of a car's speed in qualifying, using just Q3 may provide a more valid, even comparison in car speeds, since Q2 and Q1 are slightly different from Q3 in that less cars are on the track.

I think it's worth using both to try and get a better estimate and capture the "fast car" construct.

### Non-time invariant variables that might otherwise be useful
* Driver - Drivers retire, and new drivers join F1 all the time
* Circuit - Circuits are added and removed all the time
* Average Laptime - A great lap time in the past is a horrible lap time now, due to improvements in cars, driver skill, etc.
  - See [here](https://www.reddit.com/r/formula1/comments/bahin2/f1_lap_time_progression_from_all_races_19502018/)
* Pit times
  - Dependent on rule changes e.g. refuelling
  - [Trend of declining pit times as teams refined pit operations](https://statathlon.com/analysis-of-the-pit-stop-strategy-in-f1/)
* Driver total points - [Changes in the points system over the years make it difficult to compare points between past drivers with more recent ones.](https://en.wikipedia.org/wiki/Formula_One_racing)

## Method and Statistical Approach

### The data
Data on F1 races from 1950 to 2017 were obtained as part of a class on applied data science. The data seem similar to [this set from
Kaggle](https://www.kaggle.com/cjgdev/formula-1-race-data-19502017), which can be used for easy reference on what data is available.

### Modelling approach

#### Random Forest

#### Gradient Boost

#### Iteration and Model Evaluation

#### Resampling

#### PCA

#### Adjusting Model Hyperparameters
