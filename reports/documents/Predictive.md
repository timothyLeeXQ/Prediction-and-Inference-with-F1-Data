# Predictive Task - Predict Drivers that come in second between 2011 and 2017 from 2011 to 2017 using data from 1950 to 2010

>(2) [25pts] Now we move on to prediction. Fit a model using data from 1950:2010, and predict drivers that come in second place between 2011 and 2017. [Remember, this is a predictive model where variables are selected as the subset that is best at predicting the target variable and not for theoretical reasons. This means that your model should not overfit and most likely be different from the model in (1).]
>From your fitted model:
- describe your model, and explain how you selected the features that were selected
- provide statistics that show how good your model is at predicting, and how well it performed predicting second places in races between 2011 and 2017
- the most important variable in (1) is bound to also be included in your predictive model. Provide marginal effects or some metric of importance for this variable and make an explicit comparison of this value with the values that you obtained in (1). How different are they? Why are they different?

## Introduction
In our previous inferential task (see [`Inferential.md`](Inferential.md)), we sought to explain reasons why drivers came in second place. While understanding the sport at a conceptual level has its appeal, being able to predict positions is also of interest to competitors and fans. Predictive models can help constructors determine whether a driver is worth investing in by tracking their predicted positions in races with the constructor, or can help plan strategy. In fact, teams like Mercedes, which has dominated the championship in recent years, [are already considering employing predictive analytics in their races](https://www.datanami.com/2018/04/19/go-fast-and-win-the-big-data-analytics-of-f1-racing/). For fans, predictive analytics may bring an additional component to fan discussion and speculation.

Here, we build on the inferential task from before to try and build a predictive model that can predict drivers that come in second between 2011 and 2017, using data from 1950 to 2010.

## Method and Statistical Approach

### The data
Data on F1 races from 1950 to 2017 were obtained as part of a class on applied data science. The data seem similar to [this set from
Kaggle](https://www.kaggle.com/cjgdev/formula-1-race-data-19502017), which can be used for easy reference on what data is available.

### Features

It is important for features to be time invariant - otherwise, the values of features the model is trained on will be entirely different from the values of the features the model will be tested on. Hence, otherwise potentially useful features like drivers, circuits, and constructors have to be excluded. A model that we train on old data believing that Cooper cars on Brands Hatch guarantees second place would find it difficult to deal with the 2010s, where there are no Cooper cars, and no Brands Hatch race.

#### Features significant in inferential model

##### Starting Grid Position

As noted in the inferential task writeup, a higher starting grid position is likely to advantage a higher placed finish due to overtaking being more difficult than staying in the lead.

Startin grid position was found to be the most important variable in explaining why a driver came in second in the inferential task. It was important both in explaining finishing in the top 2 positions, and also in distinguishing first from second.

Thus, Starting Grid Position is likely to be useful for predicting 2nd place finishes.

##### Pit Strategy

Different strategies create different race dynamics, juggling the need to overtake with faster speeds of the car.

Pit strategy was also found to explain top 2 finishes, and in the case of a single pit strategy, was significant in differentiating first from second. Pit strategy also had some significant conditional relationships with grid position.

Pit strategy may be useful for predicting 2nd place finishes, given findings from the inferential model. However, because most non-missing pit data comes from later races, there may not be sufficient data in the training set to match against the more recent races of the test set. If this is the case, pit strategy and any other pit-stop based feature cannot be used.

##### Constructor quality

A good constructor brings a stronger pit team, better car, and better training to a race. Constructor quality was found to be a significant predictor in the inferential model for distinguishing top 2 finishes from the rest of the competitors. The quality of the constructor, operationalised here in the same way, should therefore also be predictive of second place finishes.

#### Other features

##### Driver Experience

Driver experience was not siginificant in either inferential model for top 2 finishes or distinguishing first from second. However, in the former case, it approached significance, suggesting that it may have some predictive power in determining second place positioning.

##### Driver Average Finishing Position
In discussing the lack of significance of driver experience in the inferential model, I mentioned that driver experience might not fully capture the construct of driver skill. Some drivers may be very talented and win races from the start. Driver average finishing position would capture some of this aspect of driver skill and likely hold some predictive power.

##### Constructor Average Finishing Position
In the same way, constructor average finishing position may be a better metric of constructor quality, which is already considered to be a potential predictor above.

##### Average Qualifying lap time
Qualifying lap time did not make sense to include in the inferential model as it held little theoretical value. Qualifiying lap time as a significant model term (after controlling for the grid position earned) will simply tell you "fast tends to win", which I think is safe to assume would not even satisfy a 5 year old. However, such a direct proxy of "fast car" will likely be valuable in predicting future race outcomes in the time between the Saturday and Sunday races.

The average is used because
* Eliminations mean that Q2 and Q1 data are missing for many drivers in each race
* Gets around rule changes in the past where only a single qualifying lap was used, which creates "missing" Q2 and Q1 data that cannot be simply imputed by using the max value (as you could reasonably do for missing actual Q2/Q1 data)
* For drivers where Q2 and Q1 data are present, an average provides a more robust estimate of their speed

##### Q3 Qualifying Lap Time
While the average provides more robust estimates of a car's speed in qualifying, using just Q3 may provide a more valid, even comparison in car speeds, since Q2 and Q1 are slightly different from Q3 in that less cars are on the track.

I think it's worth using both to try and get a better estimate and capture the "fast car" construct.

#### Non-time invariant variables that might otherwise be useful
* Driver - Drivers retire, and new drivers join F1 all the time
* Circuit - Circuits are added and removed all the time
* Average Laptime - A great lap time in the past is a horrible lap time now, due to improvements in cars, driver skill, etc.
  - See [here](https://www.reddit.com/r/formula1/comments/bahin2/f1_lap_time_progression_from_all_races_19502018/)
* Pit times
  - Dependent on rule changes e.g. refuelling
  - [Trend of declining pit times as teams refined pit operations](https://statathlon.com/analysis-of-the-pit-stop-strategy-in-f1/)
* Driver total points - [Changes in the points system over the years make it difficult to compare points between past drivers with more recent ones.](https://en.wikipedia.org/wiki/Formula_One_racing)

### Outcome
Out outcome variable is whether a driver comes in 2nd, or comes in any other position.

### Iteration

I use an iterative approach to increase the chances of finding a better predictive model. Several approaches will be considered and tweaked to try and arrive at the "best predictive model", referring to better predictive performance of the trained model (using data from 1950-2010) on the "test set" (data from 2011-2017). What "better predictive performance" means precisely is discussed further below, under "Model Evaluation and Percent Correct".

Here, we discuss major parameters that can be iterated over to result in a better model.

#### Choice of Predictive Algorithm

Trying different predictive modelling algorithms may result in finding a better model. Here we use random forest and gradient boost. These methods are chosen as

* Unlike with logistic regression, these models should be able to capture the differences between 1st/2nd and 2nd/everyone else, so there shouldn't be a need to run two of these like we did with logistic regression.
* They intrinsically deal with conditional relationships
* They provide feature importances, we need to do the requested comparison with the inferential model.

#### Model Hyperparameters
Many modelling algorithms have user-defined parameters that can be set. Iterating over these may yield a model with settings that provide better performance. While trying both Random Forest and Gradient Boost algorithms, we also experiment with their hyperparameters hoping to optimise model performance further.

#### Resampling Strategy

As second place is an exclusive position, awarded only to one driver in a race of about 20 drivers (usually), the prevalence of positive cases is only about 5%, creating a severe class imbalance. This may affect our models; ability to predict second place finishes effectively.

Resampling strategies may improve the ability of the model to account for this class imbalance. We iterate between a few to see if they improve model performance:

* No resampling
* Random Undersampling
* Random Oversampling
* SMOTE

#### Model Evaluation

Typical model evaluation techniques provide the model with the test data, using which, the model outputs predicted classes. In our case, that would be 2nd place, or not 2nd place. Metrics like accuracy or precision are then used to evaluate the model performance as a ratio using some combination of True Positives (Predicted 2nd, Actually 2nd), True Negatives (Predicted not 2nd, Actually not 2nd), False Positives, and False Negatives.

However, using predicted classes is not suitable for our application of predictive modelling to F1 races as **Race positions are exclusive**. [Tied positions are very rare in racing where timings are kept to milliseconds, and only happened once in F1 history](https://www.reddit.com/r/formula1/comments/5plz2c/the_only_f1_race_which_ended_in_a_dead_heat/).

This makes typical model evaluation using predicted classes unsuited as a model accepting drivers' race data as input and outputting 2nd/any other position may predict that in a race,

* there is no 2nd place
* Or more than one 2nd place

This does not reflect the truth that every race must have a 2nd place finish, as long as there are at least 2 cars in the race.

Here, I get around this problem by using probabilities of a 2nd place finish rather than predicted classes to award the predicted 2nd place position. For each race, only the driver with the highest probability of a 2nd place finish is awarded the predicted 2nd place spot.

We will use **Sensitivity (The True Positive Rate)**, to evaluate the quality of our model. This is chosen as

* In this case of position exclusivity, sensitivity is equivalent to the proportion of races where the model predicts 2nd place correctly, which makes intuitive sense for deciding whether it does a good job.
* True negatives are not considered in calculating sensitivity
  - This is important as if number 2 is exclusive and only 1 car can earn that spot, a true positive prediction also means a true negative prediction for every other car in the race, while a false positive prediction only means a false negative prediction for one other car in the race. This means that no matter the actual quality of the model, the true negative rate will be much higher than true positives, false positives, and false negatives.


## Results and Discussion
