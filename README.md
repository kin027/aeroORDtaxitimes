# AeroORDTaxi

## Overview

A Python implementation of FOLD-SE, an explainable rule-learning algorithm, used on 2025 Bureau of Transportation Statistics (BTS) to learn rules and exceptions for long taxi times for arrivals at ORD.

### Why I Built This

I'm an avgeek (aviation enthusiast) and data nerd. At the conclusion of each flight I take, I like a long taxi in from the runway to the gate if it involves a bit of a tour of the airport. However, even I would get bored (or angsty if I'm trying to make a connection) if I start making circles around an airport.

Unfortunately, that's the reality at Chicago-O'Hare International Airport (ORD), where it's [very possible to literally taxi in a circle around the terminals](https://www.reddit.com/r/unitedairlines/comments/1uup4bw/first_lap_of_the_ord_500_almost_complete_think_we/). ORD is infamous among the road warriors for its long taxi times to the point where they call it the _Tour de O'Hare_.

I was curious about the factors that could cause an arrival into ORD to have a long taxi time (defined in the program as being longer than 14 minutes), so I used the FOLD-SE algorithm to see if any rules would appear. (To be clear, this program doesn't predict flight delays, just whether a flight into ORD will have a long taxi into its arrival gate.)

### Key Insights

I wasn't that surprised by the results (which may vary as you run it). Here's one:

```
long_taxi_in(X,'True') :- crs_arr_time(X,N1), N1>1748.0, N1=<2144.0.
```

That just means that the model predicts that a flight will have a long taxi time if has a scheduled arrival time from 5:48 PM to 9:44 PM. That, of course, makes sense due to the evening banks of flights that American and United (the two carriers that make ORD a hub) have, plus the effects of any snafus earlier in the day cascading down to the evening.

Here's another:

```
long_taxi_in(X,'True') :- not month(X,'1'), not month(X,'2'), not month(X,'3'), not month(X,'10'), not op_carrier(X,'AA'), not month(X,'9'), not day_of_week(X,'Wednesday'), not op_carrier(X,'NK'), not op_carrier(X,'F9'), not month(X,'12'), not month(X,'11'), not day_of_week(X,'Thursday'), not day_of_week(X,'Tuesday'), not day_of_week(X,'Saturday'), crs_arr_time(X,N1), N1>1300.0, N1=<1748.0, distance(X,N2), N2>235.0, N2=<1730.0.
```

There's a lot to unpack here, but it basically boils down to the model predicting that a flight that: 

- Is during the summer months
- Is on a peak travel day 
- Is not on American, Spirit (RIP, I used 2025 data), or Frontier
- Is scheduled to arrive in the afternoon
- Has a flight distance (in miles) in the range (235, 1730]

will experience a long taxi time at O'Hare. This makes sense mainly because of airport congestion during busy travel times.

Here's an interesting one:

```
long_taxi_in(X,'True') :- not month(X,'1'), not month(X,'10'), not month(X,'2'), not month(X,'3'), crs_arr_time(X,N1), N1>725.0, N1=<1000.0, distance(X,N2), not(N2=<235.0), wheels_on_to_crs_arr_time(X,N3), not(N3=<12.0).
```

I'm particularly interested in the `wheels_on_to_crs_arr_time(X,N3), not(N3=<12.0)` rule. Besides the other conditions for this rule, this one means that if a flight lands at least 12 minutes ahead of its scheduled arrival time, it would have a long taxi time. This is the classic "landing early but your gate is still occupied" situation that nobody likes. When this happens at ORD, planes often just taxi to the central de-icing facility (affectionately known as the penalty box) and chill until their gate opens up (or the flight gets reassigned to a different gate). The taxi time clock is still ticking for every second a plane sits there, though.

I suppose these rules would be helpful in informing you when booking your next connection through ORD, so you can keep your taxi times down. Or, for me, I guess to experience the _Tour de O'Hare_ myself. My last taxi into a gate at ORD surprisingly took only 11 minutes. ✈️

## Getting Started

Make sure you have Python 3.13 or higher installed.

1. Clone this repository:

```bash
git clone https://github.com/kin027/aeroORDtaxi.git
cd aeroORDtaxi
```

2. Create a virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate it:
# On macOS and Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

3. Install the dependencies (you don't need to download the data tables themselves as I've already included the final ones in the repo):

```bash
pip install -r requirements.txt
```

4. Run the program!

```bash
python main.py
```

## Libraries Used

- pandas (to analyze the OTP tables, and process the data for the algorithm)

- requests (to call the server)

- cryptography

## Data Sources

I downloaded data from the Bureau of Transportation Statistics (BTS), a part of the U.S. Department of Transportation. Airlines report their on-time performance data to the BTS each month.

- [BTS Reporting Carrier On-Time Performance (1987-present) table](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD&Yv0x=D)
  - Provides detailed on-time performance info for every domestic scheduled passenger flight in the U.S.
  - Released monthly with a 2-3 month lag to ensure data quality
  - I used data from January 2025 to December 2025

## Limitations

- The data does not include a rather important feature: arrival runway
  - So that is not a possible predicate in this analysis
  - (If you land on runway 27R, you might want to get another movie ready though...)

- The OTP tables include only U.S. domestic traffic
  - So my analysis evaluates U.S. domestic traffic only

## References

- Wang & Gupta, _FOLD-R++: A Scalable Toolset for Automated Inductive Learning of Default Theories from Mixed Data_, FLOPS 2022.
- Wang & Gupta, _FOLD-SE: an efficient rule-based machine learning algorithm with scalable explainability_, PADL 2024.
- Code: https://github.com/hwd404/FOLD-R-PP and https://github.com/hwd404/foldse_client
