# =========================================
# MATPLOTLIB CHEAT SHEET + PRACTICE
# =========================================

# import plotting library
import matplotlib.pyplot as plt


# =========================================
# SAMPLE DATA
# =========================================

# x-axis values
x = [1, 2, 3, 4]

# first y-axis dataset
y = [2, 4, 6, 8]

# second y-axis dataset
y2 = [1, 3, 5, 7]


# =========================================
# BASIC LINE PLOT
# =========================================

plt.plot(
    x,              # x values
    y,              # y values

    # line color
    color="blue",

    # line style
    # "-" solid
    # "--" dashed
    # ":" dotted
    linestyle="--",

    # point markers
    # "o" circle
    # "x" x mark
    # "s" square
    marker="o",

    # line thickness
    linewidth=3,

    # name shown in legend
    label="Data 1"
)


# =========================================
# SECOND LINE
# =========================================

plt.plot(
    x,
    y2,
    color="red",
    marker="s",
    linewidth=2,
    label="Data 2"
)


# =========================================
# TITLES + LABELS
# =========================================

# graph title
plt.title("Engineering Plot", fontsize=18)

# x-axis label
plt.xlabel("Time")

# y-axis label
plt.ylabel("Speed")


# =========================================
# GRID
# =========================================

# adds background grid
plt.grid(True)


# =========================================
# LEGEND
# =========================================

# shows labels for lines
plt.legend()


# =========================================
# SAVE GRAPH
# =========================================

# saves graph as image
plt.savefig("engineering_plot.png")


# =========================================
# SHOW GRAPH
# =========================================

# displays graph window
plt.show()



# =========================================
# EXTRA CHEAT SHEET
# =========================================

# SCATTER PLOT:
# plt.scatter(x, y)

# BAR CHART:
# plt.bar(x, y)

# CHANGE COLOR:
# color="green"

# CHANGE MARKER:
# marker="x"

# CHANGE LINE STYLE:
# linestyle=":"

# CHANGE LINE WIDTH:
# linewidth=5

# BIGGER TITLE:
# fontsize=25

# FIGURE SIZE:
# plt.figure(figsize=(8,5))

# LIMIT AXIS:
# plt.xlim(0,10)
# plt.ylim(0,20)
