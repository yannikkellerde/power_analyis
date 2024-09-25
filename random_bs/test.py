import statsmodels.api as sm

data = sm.datasets.get_rdataset("dietox", "geepack").data
print(data)
