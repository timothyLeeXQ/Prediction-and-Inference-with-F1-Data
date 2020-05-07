# Inferential Task - Explaining why a driver arrives in 2nd place in races from 1950 to 2017

>(1) [25pts] Your first task is inferential. You are going to try to explain why a driver arrives in second place in a race between 1950 and 2010. Fit a model using features that make theoretical sense to describe F1 racing between 1950 and 2010. Clean the data, and transform it as necessary, including dealing with missing data. [Remember, this will almost necessarily be an overfit model where variables are selected because they make sense to explain F1 races between 1950 and 2010, and not based on algorithmic feature selection]
From your fitted model:
>- describe your model, and explain why each feature was selected
- provide statistics that show how well the model fits the data
- what is the most important variable in your model? How did you determine that?
- provide some marginal effects for the variable that you identified as the most important in the model, and interpret it in the context of F1 races: in other words, give us the story that the data is providing you about drivers that come in second place
- does it make sense to think of it as an "explanation" for drivers arriving in second place? or is it simply an association we observe in the data?


## Introduction

[F1 is big money](https://en.wikipedia.org/wiki/Formula_One#Revenue_and_profits). [The cars are expensive to develop and maintain over the course of a championship](https://www.quora.com/How-much-does-a-F1-car-cost-1), [entering a new team requires a high upfront cost](https://en.wikipedia.org/wiki/Formula_One), and [drivers aren't cheap either](https://www.autosport.com/f1/news/148862/why-the-top-f1-drivers-are-so-highly-paid). The sport also attracts a large international fanbase and deep-pocketed sponsors. Thus, there is substantial interest in understanding success in the sport among fans, drivers, constructors, and the FIA itself.

### Understanding Position #2

In motorsport, "success" is typically associated with the number 1 spot. In F1,
legends like Ayrton Senna and Michael Schumacher are celebrated for their
ability to consistently win races and championship titles. However, there are
many reasons why those involved in F1, or F1 fans, may want to understand why a
driver comes in second place.

Teams used to coming in 2nd or below are interested in 2nd place for obvious
reasons. A higher position comes with the potential of a podium finish and more
points. In theory, a championship can be delivered by just coming in second in
every race.

For dominant drivers and constructors accustomed to the chequered flag, second place is of interest as the number 1 position is never out-of-context. [Even if a first placed driver enjoys a substantial lead over the next placed car, victory is never assured](http://atlasf1.autosport.com/98/mon/wells.html). Staying ahead necessarily requires an understanding of why a driver may end up behind another.

For fans, any additional understanding behind why the race ends as it does is likely of intrinsic interest, enhancing one's understanding of the sport.

### Need for a Theory
Here, we used statistical modelling to help explain why a driver comes in 2nd place in races. However, it is not enough to just find variables associated with the 2nd place position. An association does not necessarily mean that the associated variable is a reason why something happens. The variable may also be conceptually meaningless (e.g. red car is associated with 2nd place*) or useless (e.g. 2nd best race time). Even if the variable is truly a reason why, without being able to make sense of it, we are unable to ascertain that it is a reason why, and definitely cannot "explain why" it is the case.

Hence, it is necessary to have some initial idea of why a driver ends up in second, so that we can actually gain some real insight from the statistical model and its relation to our understanding of F1. In addition, I believe that these initial ideas need to have some practical utility/insight for competitors and fans. For instance, while the average lap time of a car during a race certainly explains a driver's final position, this is obvious - the fastest driver comes in first, the next fastest comes in second, and so on. Below, I present my own simple ideas on why a driver may come in second as an informal theory.

_\* Yes of course we can understand this being meaningful as it indicates that the car is a Ferrari, but by doing that you already creating a theoretical  basis for understanding the no.2 position (constructors matter) and operationalising Ferraris as red cars, doing effectively what I say we have to above._

## An Informal Theory of Coming in Second in Formula One

### Starting Grid Position
Starting grid position is likely highly important to a driver being able to
reach number 2. I believe that the higher one's starting position, the more likely it is that a person reaches 2nd place.

A person in pole position (1st on the starting grid) only has to concede his starting position to one other car to end up in 2nd, while a person in 2nd just has to maintain that position for the race. While maintaining 2nd or dropping only to 2nd are unlikely for the whole duration of the race given the [need to pit at least once to change tires](https://en.wikipedia.org/wiki/Formula_One#Race), a higher grid position would likely mean less need to overtake to regain the 2nd spot. Overtaking is common but associated with difficulties that likely make them less likely than maintaining a position - the resistance from the car that is ahead, having to avoid crashing into the car ahead, and suboptimal aerodynamism when trailing another car in 'dirty air', among other considerations.

Even if exceptionally skilled and in an excellent car, a driver starting at position 18 is very unlikely to be able to overtake 16 drivers to claim the number 2 spot given the difficulty and disadvantages associated with overtaking.

### Team Strategy
Team strategy involves several aspects before and during the race. Some factors include:
* How many pit stops drivers intend to make during the race, and when they are made
* How much fuel cars carry (affecting speed, and number of pit stops they must make)
* Choice of tires (affects speed depending on track, and number of pit stops)

These choices may have decisive impact on one's finishing position. A driver may appear to be ahead, but could have gained that position due to a strategy that requires more pit stops. When pitting, time lost during the stop could lead the driver to lose his position.

 I believe that team strategy may have some impact on a driver's final position, particularly when conditioned on particular starting grid positions.

### Driver Skill
The skill of drivers such as Senna, Schumacher, and more recently, Hamilton, is often touted as an important factor in their finishing position. While the general interest in drivers' own ability to influence their fate mean that it should be investigated regardless, there are many credible reasons to believe driver skill really does impact one's ability to come in 2nd.

* Better drivers drive faster and stand a better chance of a higher position
* Better drivers crash less, increasing their chance of a higher position finish
* More experienced drivers have track familiarity and experience that they can exploit for overtaking
* Better drivers may influence team strategy to their advantage

Hence, some drivers may be more likely to come in second than others. We would expect these to be more experienced in F1.

### Constructor quality
Just as drivers have reputations, some constructors are known to produce better performance. Ferrari has the most constructor championship wins of any team, while teams like Williams, McLaren, and Mercedes have also won several constructor championships. Constructor championships relate to races as their drivers' finishing positions contribute points to the constructors championship.

Constructors build their own cars, manage the pit teams, and develop team strategy. Better cars, pit times, and strategy choices for a given situation all likely contribute to a driver's ability to come in second.

Hence, some constructors would be more likely to clinch the number 2 spot
compared to others. We would expect these to be 'better' constructors.

### Track
Although every track will have drivers coming in first to last, track differences may interact with other factors to influence final positioning. These are discussed below (see Interactions).

Of particular interest is the distinction between street and purpose-built tracks. Street tracks such as the Circuit de Monaco (Monaco GP) and Marina Bay Street Circuit (Singapore GP) are [more unforgiving with poorer grip and less run off areas compared to purpose-built tracks](https://en.wikipedia.org/wiki/Street_circuit).

### Race Conditions
Individual race conditions such as the weather and the presence of a safety car during the race can have significant impacts on race outcomes. This will also  interact with team strategy, which would evolve in tandem with developing race conditions.

_However, the lack of race day weather and safety car lap data prevent us from using this in our analysis, so no further discussion of these factors will be made. A future effort may seek to use a weather service API to bring in race day weather data, and more comprehensive race information could come with records of safety car deployment._

### Interactions/Conditional Effects
* Team Strategy and Starting Grid Position

Teams may opt for different strategies depending on one's starting grid position. For instance, a driver in 2nd place may choose to start with more fuel and make less pit stops, banking on rivals in 1st and 3rd to pit more often and lose crucial time. A driver in 3rd determined to improve his position may pit early and lose his initial position, hoping to gain it back on fresh tyres and no need to pit for the remaining laps.

It is hard to hypothesise beforehand which strategy works best given a particular starting grid position, but that this may have an impact on the finishing position in the race is intuitive and should be tested.

* Driver and Starting Grid position

Some drivers may be more adept at overcoming the difficulties of starting at the back of the grid. Hence, while a worse grid position is disadvantageous to any driver, the extent to which it does so may be different depending on the driver.

I expect that better drivers would be less encumbered by a lower starting grid position than worse drivers, and would have a better chance of coming in second. Better drivers would also be more likely to retain a high grid position and finish second than worse drivers.

* Driver and track

Drivers may perform far better than competitors at some tracks. For instance, from 1984-1993, [only Alain Prost and Ayrton Senna won the Monaco Grand Prix](https://en.wikipedia.org/wiki/Monaco_Grand_Prix). Particular track affinities of drivers are worthy of investigation for fan interest, and may have an impact on teams' and drivers' training strategies.

For our particular interest here in street vs purpose-built tracks, more accomplished drivers are probably more likely to perform better on street-circuits, as their greater skill would matter more when conditions are more unforgiving.

* Driver and Constructor

This connection should be apparent. A better driver will be able to draw out more performance from a better car. On the other hand, an inferior driver may not be able to use a car to its maximum potential, rendering even the best car average. Worse, a driver may even be hindered as an inability to exploit the car's design results in the design being a liability rather than advantage.

A better driver x constructor combination should then yield much higher chances of getting 2nd place compared to a mix between a talented driver and mediocre constructor, or a mediocre driver and good constructor. Of course, a weak driver and constructor combination should stand little chance of a podium finish.

### How about number 1?
Thus far, I have explained what factors, and interactions between factors, may make it more or less likely for a driver to come in second. Where apparent, I've also explained how the factor will impact a driver's chances - for instance, a higher starting grid position will give you a higher chance of coming in second. However, everything discussed thus far simply points towards better performance. What gets you to number 2 as discussed above is also likely to get you to number 1.

However, an important question when analysing who finishes second is whether the factors that make it more likely to finish second place are simply the same as coming in first, just to a lesser extent, or if there is a real difference between coming in second and coming in first. Knowing which of these two alternatives is true will advance understanding of race dynamics, which will help teams develop strategies for drivers to improve their position, and if near the top, to finish first rather than second.

Unfortunately, it is not immediately clear to me which of these two alternatives is more likely. Yet, I will say that the factors discussed above should be the same ones to distinguish first and second. Although discussed most specifically towards the number 2 position, these factors capture aspects important to affect overall race performance for all positions, not just the number two spot.

## Method and Statistical Approach

### Modelling Approach
I use logistic regression to model the impact of the variables and their interactions on the chance of finishing second. Logistic regression models the probability of a certain event, as opposed to it not happening, given changes in the values/presence or absence of certain factors. Here, we seek to model how the probability of a number 2 finish changes in relation to the factors and interactions stated above. This will help us explain why a driver finishes second.

However, just modelling the probability of finishing second implicitly assumes that finishing in any other position is qualitatively different from finishing second. As mentioned above, this might not be true - a first place finish might just be a superior application of the same formula used with a second place finish.

To get around this problem, I employ two logistic regressions. The first models the probability of ending up first or second. This should give us an idea of why someone ends up in first or second. Then, I run a second logistic regression using only data from those finishing first and second, aiming to model the probability of ending up second instead of first. This will tell us which of the factors we discussed differentiate a first place from a second place finish, and how they do so.


### Operationalising Variables
* Starting Grid Position - Range of numbers starting from pole position (1).

The operationalisation of starting grid position should be self-explanatory, since the starting grid is a ranking system of who starts where anyway.

* Driver skill - Operationalise as experience - Number of years since debut

Driver skill is operationalised as race experience as there is no external measure of driver skill. Alternative metrics like past race performance in terms of lap times and position do not produce a good metric of driver skill. Lap time is a bad metric as lap times are also heavily influenced by changes in the cars and tracks that are part of the F1 championship, particularly in this dataset that spans almost the whole history of F1. Previous race positions is a bad metric as this is also confounded by all the other factors we are using in our model, since it is effectively the same as our dependent variable. It also does not capture the competitiveness of the sport in different time periods.

* Team strategy - Number of Pit stops. Nominal variable.

While I had a lot of fun reading about and describing the nuance of F1 strategy above, we have no data on teams' tyre choices or fuel amounts. Thankfully, those factors are related to the number of pit stops, such that I think it is reasonable to summarise (and simplify) strategy as whether the driver adopted a "one-stop", "two-stop", "three-stop", and ">3-stop" strategy.

A strategy with fewer stops likely used harder tyres that wear down slower, and loaded up on more fuel to finish the race. A strategy with more stops likely used softer tyres that provide faster speed but wear down faster, and fuelled up with less fuel to further improve speed. That said, each choice speaks to a distinct strategy and shouldn't be viewed on a scale. Hence, this is a nominal variable of strategy types.

* Constructor quality - Whether the constructor has won a championship or not

Constructor quality is difficult to opreationalise, facing the same difficulties as operationalising driver skill. Because we have no data on how long a constructor has been in the business and unlike with drivers, lack of experience means nothing since the staff of a new constructor can be very experienced (see Brawn GP), we can't use experience like we do with driver skill.

We will attempt some marker of constructor quality by differentiating the best
constructors - those capable of winning the championship - from the rest. [Only
15 constructors have won the constructor's trophy in F1 history.](https://en.wikipedia.org/wiki/List_of_Formula_One_World_Constructors%27_Champions)

## Results and Discussion

### Predicting top 2 finishes
#### Results

##### Overall Model Fit and Significance
AIC: 8606.4
* McFadden R<sup>2</sup>: 0.3542071
* Model significance:  p < 0.001

##### Coefficients
![Coefficients for logistic regression of top 2 finishes](https://i.imgur.com/FciK69k.png)
* _x-es denote 95% confidence interval bands_
* _Numerical values are available in the .Rmd output, saved as HTML in `src/models/inf_regressions.html`_

##### P-values

![P-values for logistic regression of top 2 finishes](https://i.imgur.com/q9B7ubH.png)
* _Numerical values are available in the .Rmd output, saved as HTML in `src/models/inf_regressions.html`_

##### Statistically Significant Variables

Statistically significant model terms are:
* Grid position - the lower your grid position, the less likely you are to be first or second.
* Constructor quality - having a constructor that didn't win a championship before hurts your chances of finishing in the top 2.
* 1 or 2 pit stop strategies (over the baseline of not knowing the number of stops)
  - A one and 2 pit strategy is likely to provide greater likelihood of coming in first or second over the data where we have no information on pit stops (this being older races).
  - However, none of the pit strategies that we have on record are  different from each other in improving chances to a statistically significant level (seen from overlapping confidence interval bands)
* Conditional effects between grid position and 1, 2, and 3 stop pit strategies
  - A lower grid position given 1 or 2 pit stops further reduces one's chances of getting first or second, relative to given a pit stop strategy that we have no information over (this being older races).
    - This could mean that these pit strategies are more detrimental now than they were in the past for attaining first or second, but we'll need the data that we lacked to know for sure.

##### Marginal Effects of most important variable

The most important variable is likely grid position. It has the highest absolute coefficient after constructor quality (- 0.88) and pit strategy (highest coefficient for this is 1.13 for a 1 stop strategy over missing pit-stop data). However, constructor quality and pit strategies are nominal variables while grid position has a range from 1 to over 20. Looking at their coefficients (-0.35 for grid position), starting 6 places behind already has a greater impact on the odds of coming in first or second than having a bad constructor and bad pit strategy (of more than 3 stops), even though a more than 3 stop strategy does not have a significant conditional relationship with grid position.

* Single decrease in Grid position given 1-pit strategy (over missing pit strategy data) on log odds of finishing first/second
  - -0.35 [-0.38, -0.32] + -0.26 [-0.43, -0.12] ≈ **-0.62** [-0.82, -0.44] (95%CI)
* Single decrease in Grid position given 2-pit strategy (over missing pit strategy data) on log odds of finishing first/second
  - -0.35 [-0.38, -0.32] + -0.24 [-0.37, -0.13] ≈ **-0.59** [-0.75, -0.45] (95%CI)
* Single decrease in Grid position given 3-pit strategy (over missing pit strategy data) on log odds of finishing first/second
  - -0.35 [-0.38, -0.32] +  -0.21 [-0.38, -0.07] ≈ **-0.56** [-0.77, -0.39] (95%CI)
* The impact of Grid position is not conditioned on any other variables to a statistically significant degree. Hence, impact of a single decrease in grid position given any other variable on log odds of finishing first/second is
  - **-0.35** [-0.38,  -0.32] (95%CI)

#### Discussion
Overall, the model captures a decent amount of the overall variation in top 2 placings. However, most coefficients derived fail to show statistical significance. The most important variables are grid position, pit strategy, and constructor quality.

##### Grid position
Grid position is the most important variable in explaining a first or second place finish. This suggests that starting position has a strong impact on one's finishing position, and the disadvantages of being behind another car have a strong tangible impact on the race.

What is less clear is why grid position's effect is conditioned on pit strategy in the same way, regardless of the strategy adopted over a baseline of old strategies (save for 3 or more pit stops, which should be rare and unlikely to result in the winner anyway, explaining non-significance). Each led to a further decrease in the likelihood of finishing first or second with a lower grid position. One would think that some strategies would work better for higher positions than lower positions - for instance, cars at the back may benefit from less pitstops to take advantage of cars ahead pitting, while cars at the front may pit more as less cars ahead gives them the freedom to push for faster lap times while pitting more. This is a question that is unlikely to be answered without solving the problem of missing pit strategy data for the bulk of races in the dataset - the grouping of 'old strategies' likely had a diversity of different pit strategies and normally should not be considered as a single bloc.

##### Constructor quality
The impact of this variable should be uncontroversial. Better car, pit team, etc, advisors, means higher likelihood of ending up on the podium. The model affirmed this strongly with a highly statistically significant result.

##### Pit Strategy
A single and double pit strategy seemed to improve one's chances of winning first or second over baseline of missing pit stop data. This was not true for strategies that relied on more pit stops. While the coefficients for 1, 2, 3 and more than 3 stops were not significantly different from each other, the results do suggest some favour for a 1 or 2 stop pit strategy. The disadvantages of pitting may outweigh the benefits of going into the pits after a certain point.

However, it is worth noting that going into the pits more may also be the result of drivers suffering damage on the track. More data on events that happened on the track, and the specifics of pit strategies as discussed earlier, may shed more light onto how strategy impacts race performance.

##### Driver experience
Driver experience approaches significance, but does not meet it entirely, though its coefficient value is very low. In my opinion, this suggests that driver skill does matter in attaining the number 2 spot, but drivers' years since debut in the race only captures very little of the construct that we call 'driver skill'. A future effort might do a better job at operationalising this construct, perhaps with some sort of composite measure.

### Distinguishing First from Second

#### Results

##### Overall Model Fit and Significance
* McFadden R<sup>2</sup>: 0.06738338
* Model significance:  p < 0.001
##### Coefficients
![Coefficients for logistic regression of first vs second place](https://i.imgur.com/iFCunsy.png)
* _x-es denote 95% confidence interval bands_
* _Numerical values are available in the .Rmd output, saved as HTML in `src/models/inf_regressions.html`_
##### P-values
![P-values for logistic regression of first vs second place](https://i.imgur.com/vIFkzj5.png)
* _Numerical values are available in the .Rmd output, saved as HTML in `src/models/inf_regressions.html`_

##### Statistically Significant Variables

Statistically significant model terms are:
* Grid position - the lower your grid position, the more likely you are to finish second rather than first.
* 1 pit stop strategies (over the baseline of not knowing the number of stops)
  - A one pit stop strategy is likely to lower your chance of coming in second rather than first, over a baseline of observations with missing pit strategies (mostly from older races).
* Conditional effects between grid position a 1 stop pit strategy
  - A lower grid position given 1 pit stop further increases one's chances of getting second over first, relative to given a pit stop strategy that we have no information over (this being older races).

##### Marginal Effects of most important variable

Like with predicting first and second, the most important variable is likely grid position. Despite having the lowest statistically significant coefficient, grid position has a range from 1 to over 20 rather than just 0 and 1. Looking at their coefficients (0.24 for grid position), starting from 8 places behind has a greater impact on the odds of coming in second over first than having a single pit strategy, despite significant conditional relationships of grid strategy and having one pit stop.

* Single decrease in Grid position given 1-pit strategy (over missing pit strategy data) on log odds of finishing second over first
  - 0.24 [0.17, 0.32] + 0.85 [0.31, 1.54] ≈ **1.09** [0.48, 1.86] (95%CI)
* Single decrease in Grid position given any other pit strategy (over missing pit strategy data) and other variables on log odds of finishing second over first
  - **-0.24** [0.17, 0.32] (95%CI)

#### Discussion
Overall, the model captures a poor amount of the overall variation between first and 2nd place finishes. Most coefficients derived fail to show statistical significance. The most important variables are pit strategy and starting grid position. However, only a 1-pit strategy achieved statistical significance, and it is associated with a high degree of uncertainty (as seen in the large confidence interval).

##### Grid position
That grid position remains the most important variable suggests that even amongst the top 2 positions, starting further down the grid still increases the chance of ending up second instead of first. The reasons for this are unlikely to be that different from those suggested for why they affect the top 2 placing. Being stuck behind other cars that occupy space, create dirty air, and resist overtaking make it harder for you to end up first rather than second.

##### Pit Strategy

In this model, only a one-pit strategy significantly differentiates first from second place, along with its conditional interaction with starting grid position. A one pit startegy may lead one to end up second rather than first because it brings down a car's average laptime on non-pit laps, making it difficult to overtake the lead car which enjoys a lot of clean air and no resistance, and may be maintaining a substantial lead due to that. It makes sense that this effect is compounded by grid position - a lower position means more overtaking is needed, meaning that a driver faces the disadvantages of a lower average laptime more than if he had started further up the grid.

It appears that this disadvantage outweighs the disadvantage of entering the pits more than once.

##### Constructor quality and driver experience
Interestingly, these factors which were significant or close to significant in the previous model of top 2 finishes are not significant in this model of differntiating first from second. This suggests that among the top two racers, the quality of one's team and the experience under one's belt are less decisive than strategy and starting position in determining the winner.

##### Finishing Second: Just not good enough to be first, or different from first?
The model suggests that to a large extent, finishing second means being just not good enough to be first. No factor that does not differentiate the top 2 from the rest manages to differentiate first from second. Furthermore, a lower grid position has similar effects on one's chances of finishing second rather than first, and finishing in the top 2. This suggests that 2nd is just a poorer quality showing compared to first place.

## Endnote 1: Explanation or Association
Here, we have attempted to explain why a driver comes in second using a statistical model. Care was taken to use model terms that are mostly determined well before the race outcome, as temporal precedence is a necessary pre-requisite for causal rather than associative inference. Hence, we have some degree of confidence that factors like starting grid position actually explain why someone came in second, rather than just being associated with second.

However, our certainty is far from absolute.
* First, not all significant factors in our model are truly temporally precedent - pit strategy can change during a race and may be partially dependent on one's current race position. This would suggest that race position, even if not the final position, affects pit strategy, diminishing our ability to claim that pit strategy actually explains final standings instead of being only associated with it (in a vague and difficlt to explain way).
* Second,  the collinearity of many factors serve as confounds that make it difficult to claim that any one factor truly explains the second place position, independently from other factors. For instance, both constructor quality and starting grid position affect chances of a top 2 finish, but a good constructor likely also helps you get a higher grid position. To what extent then, is each factor an explanation of one's finishing position, rather than working with/through the other factor?

## Endnote 2: Future Work
1. As mentioned above, incorporating data on race conditions may be important to capture more of the variance in finishes
2. Finding ways to dissociate the problems of collinearity that confound the explanatory value of our model would be important from learning more about second place finishes in F1.
