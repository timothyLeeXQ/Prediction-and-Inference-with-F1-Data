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

##### Constructor quality

A good constructor brings a stronger pit team, better car, and better training to a race. Constructor quality was found to be a significant predictor in the inferential model for distinguishing top 2 finishes from the rest of the competitors. The quality of the constructor, operationalised here in the same way, should therefore also be predictive of second place finishes.

##### Driver Experience

Driver experience was siginificant in the inferential model for distinguishing top 2 finishes from the rest of the competitors. Like with constructor quality then, it should also be predictive of 2nd place finishes.

#### Other features

##### Driver Average Finishing Position
In discussing the lack of significance of driver experience in the inferential model, I mentioned that driver experience might not fully capture the construct of driver skill. Some drivers may be very talented and win races from the start. Driver average finishing position would capture some of this aspect of driver skill and likely hold some predictive power.

##### Constructor Average Finishing Position
In the same way, constructor average finishing position may be a better metric of constructor quality, which is already considered to be a potential predictor above.

##### Q1 Qualifying Lap Time
Qualifying lap time did not make sense to include in the inferential model as it held little theoretical value. Qualifiying lap time as a significant model term (after controlling for the grid position earned) will simply tell you "fast tends to win", which I think is safe to assume would not even satisfy a 5 year old. However, such a direct proxy of "fast car" will likely be valuable in predicting future race outcomes in the time between the Saturday and Sunday races.

Q1 is used because it is common across all drivers. Eliminations mean that Q2 and Q3 data are missing for many drivers in each race. Also, in Q2 and Q3, drivers race with fewer cars on the track and are subject to tactical decisions like what tires to use (that may need to be reused in the real race). Hence, taking an average does not measure drivers who only have Q1 data on the same playing field as those who advance to later qualifying rounds.

##### Not considered due to missing data

Different strategies create different race dynamics, juggling the need to overtake with faster speeds of the car. Pit strategy may be useful for predicting 2nd place finishes. Pit strategy was posited to be important in the inferential task but ultimately could not be used as data only existed from 2011.

However, even though the task now extends to the period where pit data is available, pit data will still be entirely absent in the training set. Hence, pit strategy and any other pit-stop based feature cannot be used.

#### Non-time invariant variables that are hence not considered
* Driver - Drivers retire, and new drivers join F1 all the time
* Circuit - Circuits are added and removed all the time
* Average Laptime - A great lap time in the past is a horrible lap time now, due to improvements in cars, driver skill, etc.
  - See [here](https://www.reddit.com/r/formula1/comments/bahin2/f1_lap_time_progression_from_all_races_19502018/)
* Pit times
  - Dependent on rule changes e.g. refuelling
  - [Trend of declining pit times as teams refined pit operations](https://statathlon.com/analysis-of-the-pit-stop-strategy-in-f1/)
  - Pit data only available after 2011
* Driver total points - [Changes in the points system over the years make it difficult to compare points between past drivers with more recent ones.](https://en.wikipedia.org/wiki/Formula_One_racing)

### Outcome
Our outcome variable is whether a driver comes in 2nd, or comes in any other position.

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

* Random Forest Hyperparameter tuning
  - Number of trees in ensemble - 1000
  - Number of features per tree - All, or square root of total number of features
  - Tree depth - 1 to 10
* Gradient Boost Hyperparameter tuning
  - Loss function - deviance
  - Learning rate - 0.3, 0.5, 0.7, 0.9
  - Number of trees in ensemble - 1000
  - Number of features per tree - All, or square root of total number of features
  - Tree depth - 1 to 10

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

We will use **Sensitivity**, to evaluate the quality of our model.

Sensitivity, the True Positive Rate, is calculated with the following formula: Sensitivity = TP/(TP+FP), where TP is the number of true positives and FP is the number of false positives.

This is chosen as in this case of position exclusivity, sensitivity is equivalent to the proportion of races where the model predicts 2nd place correctly. This makes intuitive sense for deciding whether the model does a good job.

Also, true negatives are not considered in calculating sensitivity. This is important as if number 2 is exclusive and only 1 car can earn that spot, a true positive prediction also means a true negative prediction for every other car in the race, while a false positive prediction only means a false negative prediction for one other car in the race. This means that no matter the actual quality of the model, the true negative rate will always be much higher than true positives, false positives, and false negatives.

### Unforseen circumstances
Unfortunately it was discovered that the dataset only contains qualifying lap data from 1994 onwards, meaning that it would be missing in most of the training set. Furthermore, there was a small amount of missing data after 1994, likely among drivers who failed to finish the first qualifying round.

To deal with this issue, imputation was attempted, using the rest of the features in a K-Nearest Neighbours model to fill in the missing values.

Understanding that imputation was likely to be a problem because races from earlier in F1 history were likely to be systematically different from more recent races, I decided to hedge and add the presence or absence of q1 qualifying times as a feature as another iteration parameter.

## Results and Discussion

### Model Run Results
![Precision scores of all models run](https://i.imgur.com/YbQQoeV.png)

Model performance ranged from dismal (almost 0% precision, meaning its predictions are wrong all the time) to good (20+% precision, or predicting 1 in every 4-5 races correct*). Random forest algorithms tended to perform better than gradient boost, although there were some fairly successful gradient boost models. The presence of the imputed Q1 laptimes did not seem to harm model performance, and similar performance was seen across the different resampling techniques used to address class imbalance in the training set. The most variance in performance is seen across the model hyperparameters.

_* Given that there are about 20 drivers in a modern F1 race (like those in our test set from 2011 to 2017), meaning chance performance is getting the prediction right every 20 races (5% precision), 20% or 1-in-5 races correct should be considered good._

### Best Model

#### Precision - 0.248
This precision score is similar to saying that the model is right in about 1 in 4 races. This is pretty good performance.

#### Parameters

* Algorithm - Random Forest
* Sampling technique - Random Undersampling
* Tree depth - 7
* Number of features per tree - Square root of total number of features
* Number of trees in ensemble - 1000
* Included Q1 laptime data - No

#### Feature Importance and Comparison with Inferential Model
![Feature Importances of the Best Model](https://i.imgur.com/AqrIFH7.png)

Random Forest feature importance is measured as the proportion of times which a feature is used as a split in the constituent trees.

##### Starting Grid Position
Grid position is by far the most important feature in prediction, as it is in inference. It is used in almost twice the number of splits as the next most important variable, driver average finishing position. This is noteworthy since driver average finishing position is actually directly related to the target variable (coming in second), but is still less important than grid position.

As noted in the inferential write up, starting with a high grid position confers several advantages that seem to be critical for attaining a top position, and the model seems to have picked up on that.

##### Driver skill
Driver experience, which was found to be associated with a top 2 position to a statistically significant degree in the inferential model, also plays a role in the predictive model. However, that role is small, and may be attributed at least in part to the absence of other more important features in some of the trees that made up the random forest.

However, driver average finishing position was found to be a very important predictor. This supports, but far from asserts, my earlier suspicion that a better opreationalisation of driver skill may show driver skill to be more important to one's finishing position. This may also be due to average finishing position being directly related to a second place finish as well.

##### Constructor quality
Summing the dummy coded constructor quality features gives a relative importance greater than driver experience. This comparison of importance is similar to the inferential logistic regression modelling top two finishes, where constructor quality is more important than driver experience but less important than grid position, as measured by standardised coefficients.

However, constructor average finishing position shows a weaker relative importance to driver average finishing position. This suggests that the skill of the constructor is less crucial than the skill of the driver in securing second place (with this not being captured in the winning constructor and driver experience operationalisations of driver and constructor skill that were the sole measures of these used in the inferential model).

However, this may also be due to average finishing position being directly related to a second place finish more so than the constructors' average finishing position. It may also reflect changes in constructors' quality over the years that render it a weaker predictor of future performance.

## Endnote: Future Work

1. Iterate with different algorithms
2. Include other likely meaningful features if they can be obtained
  - Pit strategy and other aspects of team strategy
  - Weather and race conditions
  - Better operationalisation of driver and constructor skills
3. Run model with full qualifying laptime data if it can be obtained
