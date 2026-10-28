# AeroORDTaxiTimes

## Overview

A Python implementation of FOLD-SE, a rule-based explainable machine learning algorithm, used on 2025 Bureau of Transportation Statistics (BTS) data to learn rules that predict whether a flight into ORD will have a long taxi to its arrival gate after it lands.

(To be clear, this program doesn't predict flight delays for arrivals into ORD. A flight that lands early may still have a long taxi to its arrival gate.)

### Why I Built This

I'm an avgeek (aviation enthusiast) and data nerd. At the conclusion of each flight I take, I like a long taxi in from the runway to the gate if it involves a bit of a tour of the airport. However, even I would get bored (or angsty if I'm trying to make a connection) if I start making circles around an airport.

Unfortunately, that's the reality at Chicago-O'Hare International Airport (ORD), where it's very possible to [literally taxi in a circle around the terminals](https://www.reddit.com/r/unitedairlines/comments/1w146zy/satisfying_full_loop_and_then_some_getting_your/). ORD is infamous among the frequent flyers for its ridiculously long taxi times.

So, when I first learned about the FOLD-SE algorithm, I knew I had to try it out on the BTS's Airline On-Time Performance (OTP) data to explore the combination of factors that could cause an arrival into ORD to have a long taxi, **which I defined in the program as being greater than 14 minutes**.

### Key Insights

The learned rules are expressed in Answer Set Programming (ASP) / Prolog syntax, which reads as IF–THEN logic with exceptions:

- `:-` means "IF" (the left side is true if the conditions on the right are met)

- `,` means "AND"

- `not` represents an exception or negation

The features in the OTP data include a flight's:

- Departure month (`month`)

- Departure day of the week (`day_of_week`)

- Operating carrier (`op_carrier`)

- Gate departure delay, minutes (`dep_delay`)

- Scheduled gate arrival time as an integer (`crs_arr_time`)

- Distance, miles (`distance`)

- Duration from wheels on to scheduled gate arrival time, minutes (`wheels_on_to_crs_arr_time`)

As for the rules themselves, I wasn't actually that surprised by them. They may vary for you each time you run it as the training and testing data are randomly split upon each run. Here's one:

```prolog
long_taxi_in(X,'True') :- crs_arr_time(X,N1), N1>1748.0, N1=<2144.0.
```

So FOLD-SE predicts that a flight will have a long taxi time if it has a scheduled gate arrival time from 5:48 PM to 9:44 PM. That, of course, makes sense due to the evening banks of flights that American and United (the two carriers that make ORD a hub) have, plus the effects of any snafus earlier in the day cascading down to the evening.

Here's another:

```prolog
long_taxi_in(X,'True') :- not month(X,'1'), not month(X,'10'), not month(X,'2'), not month(X,'3'), crs_arr_time(X,N1), N1>725.0, N1=<1000.0, distance(X,N2), not(N2=<235.0), wheels_on_to_crs_arr_time(X,N3), not(N3=<12.0).
```

I'm particularly interested in the `wheels_on_to_crs_arr_time(X,N3), not(N3=<12.0)` predicate for this rule. It means that if a flight lands more than 12 minutes ahead of its scheduled gate arrival time and the other predicates also hold, then FOLD-SE predicts that it would have a long taxi time. This is the classic "you landed early but your gate is not ready yet" dance that nobody likes. When this happens at O'Hare, inbound aircraft often get sent to the central de-icing facility, affectionately known as the "penalty box", to wait it out while the taxi time clock continues to tick away.

There are more rules beyond this, and I suppose they would be helpful in informing you when booking your next connection through ORD, so you can keep your taxi times down. Or, if you're like me, to maximize your chance of experiencing the _Tour de O'Hare_ yourself, because a taxi in to a gate there can be a big ORDeal... ✈️

## Getting Started

Have Python 3.13+ installed on your machine. You do not need to download any data, as I've already included the relevant data in this repo.

```bash
# Clone the repo
git clone https://github.com/kin027/aeroORDtaxitimes.git
cd aeroORDtaxitimes

# Create the virtual environment
python -m venv venv

# Activate it on macOS and Linux
source venv/bin/activate

# Activate it on Windows (Command Prompt)
venv\Scripts\activate

# Activate it on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the program
python main.py
```

## Libraries Used

- pandas (to analyze the OTP tables and process the data for FOLD-SE)

- requests (for FOLD-SE)

- cryptography (for FOLD-SE)

## Data Sources

- [BTS, Airline On-Time Performance Data, Reporting Carrier On-Time Performance (1987-present)](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD&Yv0x=D)
  - Contains scheduled and actual departure and arrival times reported by certified U.S. air carriers that account for at least half of one percent of domestic scheduled passenger revenues
  - Released monthly with a 2-3 month lag
  - I used data from January 2025 to December 2025

## Limitations

- The data does not include two important features: arrival runway and arrival terminal
  - So those cannot be generated as predicates
  - (If you land on runway 9L/27R, you might want to get another movie ready, though...)

## Future Improvement Plans

- Finding a reliable dataset to join on the tail numbers column in the OTP data for ICAO aircraft types to use that as a feature

- Including some graphs about ORD taxi times just for kicks

## References

- Wang & Gupta, _FOLD-SE: an efficient rule-based machine learning algorithm with scalable explainability_, PADL 2024.
  - Code: https://github.com/hwd404/foldse_client
