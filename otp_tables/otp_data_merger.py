import pandas as pd
import glob
import os

# Get absolute path of the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find files in that same folder
file_pattern = os.path.join(script_dir, "*2025.csv")
file_list = glob.glob(file_pattern)

processed_chunks = []
otp_df = pd.DataFrame()

for otp_file in file_list:
    # Read OTP monthly file
    otp_monthly_df = pd.read_csv(otp_file).rename(columns=str.lower) # type: ignore

    # Keep only rows where destination is ORD and flights are not canceled or diverted
    otp_monthly_df = otp_monthly_df[(otp_monthly_df["dest"] == "ORD") & (otp_monthly_df["cancelled"] == 0) &
    (otp_monthly_df["diverted"] == 0)]

    # Convert DAY_OF_WEEK from number (e.g. 1) to actual day (e.g. "Monday")
    day_of_week_map = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }

    otp_monthly_df["day_of_week"] = otp_monthly_df["day_of_week"].map(day_of_week_map)

    # Create new column for duration from wheels_on to crs_arr_time (wheels_on_TO_crs_arr_time)
    crs_mins_to_midnight = (otp_monthly_df["crs_arr_time"] // 100) * 60 + otp_monthly_df["crs_arr_time"] % 100
    wheels_on_mins_to_midnight = (otp_monthly_df["wheels_on"] // 100) * 60 + otp_monthly_df["wheels_on"] % 100

    otp_monthly_df["wheels_on_to_crs_arr_time"] = (crs_mins_to_midnight - wheels_on_mins_to_midnight + 720) % 1440 - 720

    # Create new column for taxi_in > 14 minutes (long_taxi_in)
    otp_monthly_df["long_taxi_in"] = otp_monthly_df["taxi_in"] > 14

    # Drop dest, cancelled, diverted, taxi_in, wheels_on, year columns
    otp_monthly_df = otp_monthly_df.drop(columns=["tail_num", "dest", "cancelled", "diverted", "taxi_in", "wheels_on",
                                                  "year"])

    # Append df to list of processed chunks
    processed_chunks.append(otp_monthly_df)

# Concatenate all monthly chunks
otp_df = pd.concat(processed_chunks)

# Drop NA rows
otp_df = otp_df.dropna()

# Set the output path for the final csv
output_path = os.path.join(script_dir, "..", "data", "combined_otp.csv")

# Output final_otp.csv
otp_df.to_csv(output_path, index=False)