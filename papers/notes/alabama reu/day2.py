# =========================================
# DAY 2 — PANDAS PRACTICE
# =========================================

# import libraries
import pandas as pd
import matplotlib.pyplot as plt


# =========================================
# CREATE SAMPLE DATA
# =========================================

data = {

    # time values
    "time": [1,2,3,4,5],

    # vehicle speed
    "speed": [10,15,20,18,25],

    # following distance
    "distance": [5,7,10,11,15],

    # whether ACC is active
    "acc_active": [0,1,1,1,0]
}


# convert dictionary into dataframe (table)
df = pd.DataFrame(data)


# =========================================
# VIEW DATA
# =========================================

# print full dataframe
print(df)

# first 5 rows
print(df.head())


# =========================================
# VIEW COLUMN NAMES
# =========================================

print(df.columns)


# =========================================
# ACCESS SPECIFIC COLUMN
# =========================================

print(df["speed"])


# =========================================
# BASIC STATISTICS
# =========================================

# average speed
print("Average Speed:")
print(df["speed"].mean())

# maximum speed
print("Max Speed:")
print(df["speed"].max())


# =========================================
# FILTER DATA
# =========================================

# show only rows where speed > 15
fast_data = df[df["speed"] > 15]

print("Fast Data:")
print(fast_data)


# =========================================
# CREATE GRAPH
# =========================================

plt.plot(
    df["time"],
    df["speed"],
    marker="o",
    linewidth=3,
    label="Vehicle Speed"
)

plt.title("Vehicle Speed Over Time")

plt.xlabel("Time")

plt.ylabel("Speed")

plt.grid(True)

plt.legend()

plt.show()

plt.plot(df["time"], df["speed"], label="Speed")
plt.plot(df["time"], df["distance"], label="Distance")

plt.legend()
plt.show()