# Inferential Task - Explaining why a driver arrives in 2nd place in races from 1950 to 2017

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

Hence, it is necessary to have some initial idea of why a driver ends up in second, so that we can actually gain some real insight from the statistical model and its relation to our understanding of F1. I present my own simple ideas below, as an informal theory.

_\* Yes of course we can understand this being meaningful as it indicates that the car is a Ferrari, but by doing that you already creating a theoretical  basis for understanding the no.2 position (constructors matter) and operationalising Ferraris as red cars, doing effectively what I say we have to above._

## An Informal Theory of Coming in Second in Formula One

### Variables of Interest
<Explain why these variables and not others - mention the practical value of the theory as improving pre-race commentary/thinking.

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

### Driver Skill
The skill of drivers such as Senna, Schumacher, and more recently, Hamilton, is often touted as an important factor in their finishing position. While the general interest in drivers' own ability to influence their fate mean that it should be investigated regardless, there are many credible reasons to believe driver skill really does impact one's ability to come in 2nd.

* Better drivers drive faster and stand a better chance of a higher position
* Better drivers crash less, increasing their chance of a higher position finish
* More experienced drivers have track familiarity and experience that they can exploit for overtaking
* Better drivers may influence team strategy to their advantage

### Constructor quality
Just as drivers have reputations, some constructors are known to produce better performance. Ferrari has the most constructor championship wins of any team, while teams like Williams, McLaren, and Mercedes have also won several constructor championships. Constructor championships relate to races as their drivers' finishing positions contribute points to the constructors championship.

Constructors build their own cars, manage the pit teams, and develop team strategy. Better cars, pit times, and strategy choices for a given situation all likely contribute to a driver's ability to come in second.

### Track


### Race Conditions
Safety car
Weather

### Interactions
* Team Strategy and Starting Grid Position
Teams may opt for different strategies depending on one's starting grid position. For instance, a driver in 2nd place may choose to start with more fuel and make less pit stops, banking on rivals in 1st and 3rd to pit more often and lose crucial time. A driver in 3rd determined to improve his position may pit early and lose his initial position, hoping to gain it back on fresh tyres and no need to pit for the remaining laps.

## Statistical Approach

## Method

## Results and Discussion

## Future Work


>(1) [25pts] Your first task is inferential. You are going to try to explain why a driver arrives in second place in a race between 1950 and 2010. Fit a model using features that make theoretical sense to describe F1 racing between 1950 and 2010. Clean the data, and transform it as necessary, including dealing with missing data. [Remember, this will almost necessarily be an overfit model where variables are selected because they make sense to explain F1 races between 1950 and 2010, and not based on algorithmic feature selection]
From your fitted model:
>- describe your model, and explain why each feature was selected
- provide statistics that show how well the model fits the data
- what is the most important variable in your model? How did you determine that?
- provide some marginal effects for the variable that you identified as the most important in the model, and interpret it in the context of F1 races: in other words, give us the story that the data is providing you about drivers that come in second place
- does it make sense to think of it as an "explanation" for drivers arriving in second place? or is it simply an association we observe in the data?
