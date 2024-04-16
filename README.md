# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:

### STEP 1:
Read the given Data.
### STEP 2:
Clean the Data Set using Data Cleaning Process.
### STEP 3:
Apply Feature Encoding for the feature in the data set.
### STEP 4:
Apply Feature Transformation for the feature in the data set.
### STEP 5:
Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by : LAAKSHIT D
### Reg No : 212222230071

```python

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![Screenshot 2024-04-16 111250](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/aa1d9d6b-6e3a-4448-a105-80693ce73aac)

### ORDINAL ENCODER

```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![Screenshot 2024-04-16 111400](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/bacc0a4e-1baa-4205-8c0e-20a1933b8877)

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![Screenshot 2024-04-16 111439](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/3f58f253-5b96-4fba-8838-2adc6df39427)

### LABEL ENCODER

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![Screenshot 2024-04-16 111847](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/09a6c100-6886-4391-8ad1-6d755f8588c2)

### OneHotEncoder

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/b13b4200-2403-4d37-8641-60376a4e1f1d)

```py
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/1bbec935-2f00-4bb1-b31f-9d5ffc34578d)

### BinaryEncoder

```py
pip install --upgrade category_encoders
```
```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/c75aa6d5-4992-4f49-a3be-45a0837f9d9d)

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/c52afc71-f4c5-4a2c-835a-de304d68e9bc)

```py
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/3523348b-e1fb-4e20-b136-ca5d9a3a6fed)

### TargetEncoder

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/9044f624-59af-4264-b1a8-e87221f47f56)

### FEATURE TRANSFORMATION

```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/90941400-3223-443e-97cd-5f93e4bdf557)

```py
df.skew()
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/e489eb51-0f40-428b-a95a-c822d16942d4)

```py
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/a083b3da-d864-4bcf-852d-7b353cb74353)


```py
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/0b92a9a5-b9c0-47f1-be4f-a12ae70f9edc)

```py
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/8c2b4957-e284-4610-a604-1e534fd601f1)

```py
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/laakshit-D/EXNO-3-DS/assets/119559976/dd61b053-3b11-47a0-9e6d-5c4ef11c9d96)

```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![319861757-fd614327-c87a-46fa-883e-0dce5e136cce](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/2532f626-ec68-4ec9-b417-6e6d98e79f57)

```py
df.skew()
```

![319861793-805e86cf-f9f5-4e20-a2ea-6b7328bc6443](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/8409f0fd-1134-41f1-9d54-6c2dc904edc2)

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![319861825-6ce34d13-c097-4706-b409-4cf80b55e39e](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/66035a7a-5708-430e-a2a3-0c0191dee49b)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![319861866-55399f4f-9c8c-4be0-b747-31da55de4387](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/87966f02-982b-4c43-bdf0-b9ee9992dbbe)

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![319861899-54adc02b-d562-442c-be3a-466037a3c753](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/4f686d45-2277-46ac-a498-241f17a297f3)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![319861945-0388b0c8-b211-45ad-a901-321712a8c922](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/31e9fb93-4a26-45b3-a4c0-1a230b83b9ac)

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![319861982-c82b4630-bfe9-4d2b-9250-bb22d423b2f4](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/633826a7-3484-481b-a03b-4ca033021613)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![319862041-e86811e2-d338-4d47-8285-9387cf87c191](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/0a23c373-4be7-4f31-ac5a-cb697781b896)


## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
