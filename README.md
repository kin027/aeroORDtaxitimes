# AeroORDTaxiTimes

## Overview

A Python implementation of FOLD-SE, an explainable rule-learning algorithm, used on 2025 Bureau of Transportation Statistics (BTS) data to learn rules and exceptions that predict whether a flight into ORD will have a long taxi time to its arrival gate.

(To be clear, this program doesn't predict flight delays, just whether a flight into ORD will have a long taxi into its arrival gate.)

### Why I Built This

I'm an avgeek (aviation enthusiast) and data nerd. At the conclusion of each flight I take, I like a long taxi in from the runway to the gate if it involves a bit of a tour of the airport. However, even I would get bored (or angsty if I'm trying to make a connection) if I start making circles around an airport.

Unfortunately, that's the reality at Chicago-O'Hare International Airport (ORD), where it's very possible to [literally taxi in a circle around the terminals](https://www.reddit.com/r/unitedairlines/comments/1uup4bw/first_lap_of_the_ord_500_almost_complete_think_we/). ORD is infamous among the frequent flyers for its ridiculously long taxi times.

So, when I first learned about the FOLD-SE algorithm, I knew I had to try it out on the BTS's airline on-time performance data to explore the combination of factors that could cause an arrival into ORD to have a long taxi time, **which I defined in the program as being longer than 14 minutes**.

### Key Insights

I wasn't actually that surprised by the rules that were generated (which may vary as you run it). Here's one:

```
long_taxi_in(X,'True') :- crs_arr_time(X,N1), N1>1748.0, N1=<2144.0.
```

That just means that FOLD-SE predicts that a flight will have a long taxi time if it has a scheduled arrival time from 5:48 PM to 9:44 PM. That, of course, makes sense due to the evening banks of flights that American and United (the two carriers that make ORD a hub) have, plus the effects of any snafus earlier in the day cascading down to the evening.

Here's another:

```
long_taxi_in(X,'True') :- not month(X,'1'), not month(X,'2'), not month(X,'3'), not month(X,'10'), not op_carrier(X,'AA'), not month(X,'9'), not day_of_week(X,'Wednesday'), not op_carrier(X,'NK'), not op_carrier(X,'F9'), not month(X,'12'), not month(X,'11'), not day_of_week(X,'Thursday'), not day_of_week(X,'Tuesday'), not day_of_week(X,'Saturday'), crs_arr_time(X,N1), N1>1300.0, N1=<1748.0, distance(X,N2), N2>235.0, N2=<1730.0.
```

There's a lot to unpack here, but it basically boils down to FOLD-SE predicting that if a flight meets all of the following conditions...

- Is during the summer months
- Is on a peak travel day 
- Is not on American, Spirit (RIP, I used 2025 data), or Frontier
- Is scheduled to arrive in the afternoon
- Is 235 to 1730 miles in distance

...it will experience a long taxi time at O'Hare. This makes sense because of airport congestion during busy travel times (although I was a bit surprised to see that American was an exception).

Here's an intriguing one:

```
long_taxi_in(X,'True') :- not month(X,'1'), not month(X,'10'), not month(X,'2'), not month(X,'3'), crs_arr_time(X,N1), N1>725.0, N1=<1000.0, distance(X,N2), not(N2=<235.0), wheels_on_to_crs_arr_time(X,N3), not(N3=<12.0).
```

I'm particularly interested in the `wheels_on_to_crs_arr_time(X,N3), not(N3=<12.0)` rule. Besides the other predicates for this rule, this one means that if a flight lands at least 12 minutes ahead of its scheduled arrival time, it would have a long taxi time. This is the classic "landing early but your gate is still occupied" situation that nobody likes. When this happens at O'Hare, inbound flights often get sent to the central de-icing facility, affectionately known as the "penalty box", to chill until their gate opens or gets reassigned. But while the aircraft sits in timeout, the taxi-in clock keeps ticking away.

I suppose these rules would be helpful in informing you when booking your next connection through ORD, so you can keep your taxi times down. Or, if you're like me, to maximize your chance of experiencing the _Tour de O'Hare_ yourself. (My last taxi in at ORD took only 11 minutes. I feel robbed, but also my flight didn't fit any of the rules FOLD-SE generated.) ✈️

## Getting Started

<details>
<summary>Setup and execution instructions</summary>

Have Python 3.13+ installed on your machine. You do not need to download any data, as I've already included the relevant data in this repo.

```bash
git clone https://github.com/kin027/aeroORDtaxitimes.git
cd aeroORDtaxitimes

# Create the virtual environment
python -m venv venv

# Activate it

# On macOS and Linux
source venv/bin/activate

# On Windows (Command Prompt)
venv\Scripts\activate

# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the program
python main.py
```
</details>

## Libraries Used

- pandas (to analyze the OTP tables, and process the data for the algorithm)

- requests (for FOLD-SE)

- cryptography (for FOLD-SE)

## Data Sources

I downloaded data from the Bureau of Transportation Statistics (BTS), a part of the U.S. Department of Transportation. Airlines report their on-time performance data to the BTS each month.

- [BTS Reporting Carrier On-Time Performance (1987-present) table](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD&Yv0x=D)
  - Provides detailed on-time performance info for every domestic scheduled passenger flight in the U.S.
  - Released monthly with a 2-3 month lag to ensure data quality
  - I included the Year, Month, DayOfWeek, IATA_CODE_Reporting_Airline, Tail_Number, Dest, DepDelay, WheelsOn, TaxiIn, CRSArrTime, Cancelled, Diverted, and Distance columns and used data from January 2025 to December 2025

## Limitations

- The data lacks a rather important feature: arrival runway
  - So arrival runway cannot be generated as an explanatory predicate
  - (If you land on runway 9L/27R, you might want to get another movie ready, though...)

- The OTP tables include only U.S. domestic flights
  - So my analysis evaluates U.S. domestic flights only

## References

- Wang & Gupta, _FOLD-SE: an efficient rule-based machine learning algorithm with scalable explainability_, PADL 2024.
  - Code: https://github.com/hwd404/foldse_client
