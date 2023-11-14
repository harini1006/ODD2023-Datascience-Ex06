# ODD2023-Datascience-Ex06
### AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

### ALGORITHM:
1.Read the given Data
2.Perform Data cleaning process on the dataset.
3.Apply Feature Transformation techniques to all the features of the data set
4 Analyse the transformed features

### CODE AND OUTPUT:
```
DEVELOPED BY:HARINI V
REGISTER NO: 212222230044
```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

```
![image](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/27487690-b597-42a1-aab8-36c5e73fa862)
```
df.info()

```
![image](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/eefdc89f-7b5b-4ff9-b420-1536815a7d1b)
```
df.skew()

```
![image](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/5b4b998d-1e92-47a1-8dc3-c526e64f653a)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/4644e23e-f713-4612-ab26-69705e520839)
```

np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/b1b68504-71a7-44f4-a490-1d3c3c53ddb3)

```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/118807d2-19b6-44e6-97c2-10423e878e39)
```
np.square(df['Highly Positive Skew'])
```
![image](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/f1947f58-f51f-48d8-b6c2-6d153998be75)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![275121027-b87a2380-5eda-4931-93ad-a07323f42805](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/7c225756-aaab-4659-a26b-4748df2d9546)
```
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![275121050-155e1e88-ae20-42e9-9886-4a5f217b6e46](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/15a2330c-5e50-4375-8d17-846e7a862508)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[['Moderate Negative Skew']])
df

```
![275121061-52453472-ea35-4a76-a220-76bab9f7d192](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/6f9f6edf-f777-4bd9-bfb0-3ea67066562a)
```

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![275121098-a8c6beb5-60fc-44ea-bccd-adeca59312d9](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/d9ff0d67-a1a4-4fb9-adbc-fbf5c26dffc8)

```

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()
```

![275121107-9c19d20a-6038-4c11-960c-62239f28fa13](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/ab757b8d-4f16-49fc-824e-aa36ba174fad)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[['Highly Negative Skew']])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![275121119-589a6c8d-a0a7-4aea-b1f0-9a0615e78621](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/065cd27d-4df1-4f3d-9794-6e1b50dc5f3a)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![275121128-c319964c-cdea-4739-af3b-f6fee56f3039](https://github.com/Aravindsamy04/ODD2023-Datascience-Ex06/assets/113497037/85e70ccb-1800-45d8-bb2c-bc8fd214aee2)


### RESULT:
Thus feature transformation is done for the given dataset.

