import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# User Input Form : ['TIME OCC', 'AREA', 'Vict Age', 'Vict Sex', 'Vict Descent']
def User_Input_Preprocessing(CrimeSet):
    
    # Tuning time data to two digit (0~23)
    def convert_integer(row):
        integer = row['TIME OCC']
        if integer >= 100:
            return integer // 100
        elif integer >= 10:
            return integer // 10
        else:
            return 0
        
    # Set time data so two digit (0~23)
    CrimeSet['TIME OCC'] = CrimeSet.apply(convert_integer, axis=1)
    
    CrimeSet = CrimeSet[CrimeSet['Vict Sex'] != 'H']
    # Classify descent as Black, White, Asian, Other
    asian=['A','C','D','F','J','K','L','V','Z']
    CrimeSet['Vict Descent'] = CrimeSet['Vict Descent'].replace(asian, 'Asian')
    other=['G','H','P','U','I','O','S','X']
    CrimeSet['Vict Descent'] = CrimeSet['Vict Descent'].replace(other, 'Other')

    # Set NAN Descent as Other
    CrimeSet.dropna(subset = ['Vict Descent'], inplace = True)
    # Set NAN Sex as X
    CrimeSet['Vict Sex']=CrimeSet['Vict Sex'].replace(np.nan,'X')

    # Make Freq F by Hour : felony ratio per hour
    # Order setted as ratio of felony
    order = [12, 0, 11, 10, 8, 6, 9, 13, 7, 14, 15, 16, 1, 18, 17, 20, 19, 21, 22, 2, 23, 5, 3, 4]
    CrimeSet['Freq F By Hour'] = CrimeSet['TIME OCC'].map(lambda x: order.index(x))

    # Make Freq F by AREA : felony ratio per area
    # Order setted as ratio of felony
    order = [1, 6, 7, 8, 21, 3, 17, 16, 15, 19, 9, 2, 14, 10, 11, 20, 18, 5, 13, 12, 4]
    CrimeSet['Freq F By AREA'] = CrimeSet['AREA'].map(lambda x: order.index(x))

    # Make Freq F by Victim Sex : felony ratio per sex of victim
    # Order setted as ratio of felony
    order = ['F','M','X']
    CrimeSet['Freq F By Sex'] = CrimeSet['Vict Sex'].map(lambda x: order.index(x))

    # Make Freq F by Descent : felony ratio per descent of victim
    # Order setted as ratio of felony
    order = ['Asian','B','W','Other']
    CrimeSet['Freq F By Descent'] = CrimeSet['Vict Descent'].map(lambda x: order.index(x))

    # Data of Age of victim into integer
    CrimeSet['Vict Age'] = CrimeSet['Vict Age'].astype(int)
    # Set age in unit of ten
    CrimeSet['Vict Age'] = CrimeSet['Vict Age'].apply(lambda x: math.floor(x/10)*10)
    # Clear -10
    CrimeSet = CrimeSet[CrimeSet['Vict Age'] != -10]
    # Clear 0
    CrimeSet = CrimeSet[CrimeSet['Vict Age'] != 0]

    # Make Freq F by Age : felony raio per Age of victim
    # Order setted as ratio of felony
    order = [10,70,60,50,80,90,40,30,20]
    CrimeSet['Freq F By Age'] = CrimeSet['Vict Age'].map(lambda x: order.index(x))
    drop_columns=['Vict Descent', 'Vict Sex', 'TIME OCC', 'AREA', 'Vict Age']
    CrimeSet.drop(drop_columns,axis=1,inplace=True)
    
    print("User Input Preprocessing Done!")
    return CrimeSet


def For_Full_Data_Model(CrimeSet):
    # Tuning time data to two digit (0~23)
    def convert_integer(row):
        integer = row['TIME OCC']
        if integer >= 100:
            return integer // 100
        elif integer >= 10:
            return integer // 10
        else:
            return 0
        
    # Set time data so two digit (0~23)
    CrimeSet['TIME OCC'] = CrimeSet.apply(convert_integer, axis=1)

    # Droping useles colmns
    drop_columns=['Date Rptd','DR_NO','Rpt Dist No','Part 1-2','Premis Cd','Premis Desc','Status Desc','Cross Street','LAT','LON', 'LOCATION']
    CrimeSet.drop(drop_columns, axis=1, inplace=True)

    # Set Regular expression pattern for 'DATE OCC'(MM/DD/YYYY HH:MM:SS APM)
    pattern = r'\s\d{2}:\d{2}:\d{2}\s[APM]+' # Erase off without MM/DD/YYYY

    # Fit the pattern into 'DATE OCC"
    CrimeSet['DATE OCC'] = CrimeSet['DATE OCC'].str.replace(pattern, '', regex=True)

    # Check null data in Victim Sex, Victim Descent
    crime_check = CrimeSet.loc[:,['Vict Sex','Vict Descent']]
    crime_check = crime_check[pd.isna(crime_check['Vict Sex'])]
    # print(crime_check['Vict Sex'].isna().sum())
    # print(crime_check['Vict Descent'].isna().sum())
    crime_outlier=crime_check[crime_check['Vict Sex'].isna()&~crime_check['Vict Descent'].isna()]
    # print(crime_outlier)
    # print(CrimeSet['Vict Descent'].value_counts())

    # Drop H from Victim Sex colnm
    CrimeSet = CrimeSet[CrimeSet['Vict Sex'] != 'H']

    # Classify descent as Black, White, Asian, Other
    asian=['A','C','D','F','J','K','L','V','Z']
    CrimeSet['Vict Descent'] = CrimeSet['Vict Descent'].replace(asian, 'Asian')
    other=['G','H','P','U','I','O','S','X']
    CrimeSet['Vict Descent'] = CrimeSet['Vict Descent'].replace(other, 'Other')

    # Set NAN Descent as Other
    CrimeSet.dropna(subset=['Vict Descent'], inplace=True)
    # Set NAN Sex as X
    CrimeSet['Vict Sex']=CrimeSet['Vict Sex'].replace(np.nan,'X')
    
    
    CrimeSet['Weapon Or Not'] = np.where(~CrimeSet['Weapon Used Cd'].isna(), 1, 0)
    CrimeSet['Weapon Or Not'].value_counts()
    CrimeSet.drop('Weapon Desc',axis=1,inplace=True)
    CrimeSet.drop('Weapon Used Cd',axis=1,inplace=True)

    # Critical Crime Cd
    felony=[230,231,210,310,740,121,341,122,910,350,920,648,
        113,653,950,865,354,649,237,922,235,814,760,870,
        510,331,236,930,815,668,761,940,662,320,343,522,860,
        753,520,220,622,421,820,812,435,822,810,110,251,
        755,434,250,850,821,805,921,756,840,806,948]
    # Less critical Crime Cd
    misdemeanor_among_felony=[330,341,350,950,354,930,668,940,662,343,520,220,
                            421,820,434,850,821,805,948]
    # felony = felony - misdemeanor_among_felony
    felony = [val for val in felony if val not in misdemeanor_among_felony]

    # Check duplication
    duplicates = set(x for x in felony if felony.count(x) > 1)

    # felony = 2 , misdemeanor_among_felony = 1 , others 0
    CrimeSet['Crime Class'] = CrimeSet['Crm Cd'].apply(lambda x: 2 if x in felony else (1 if x in misdemeanor_among_felony else 0))

    # Count Crime using Crm Cd 1,2,3,4
    CrimeSet['Crime Count'] = 4 - CrimeSet[['Crm Cd 1','Crm Cd 2','Crm Cd 3','Crm Cd 4']].isna().sum(axis=1)
    CrimeSet['Crime Count'].value_counts()
    drop_columns=['Crm Cd 1','Crm Cd 2','Crm Cd 3','Crm Cd 4']
    CrimeSet.drop(drop_columns,axis=1,inplace=True)

    # Make Average CC By TA : Mean of Crime count in grouped of Time and area
    grouped_data = CrimeSet.groupby(['TIME OCC', 'AREA'])
    mean_crime_count = grouped_data['Crime Count'].mean()
    CrimeSet['Average CC By TA'] = CrimeSet.apply(lambda row: mean_crime_count.get((row['TIME OCC'], row['AREA']), 0), axis=1)
    CrimeSet['Average CC By TA']=CrimeSet['Average CC By TA'].round(decimals=3)

    # Make Average CClass By TA :  Mean of Crime class in grouped of Time and area
    grouped_data = CrimeSet.groupby(['TIME OCC', 'AREA'])
    mean_crime_class = grouped_data['Crime Class'].mean()
    CrimeSet['Average CClass By TA'] = CrimeSet.apply(lambda row: mean_crime_class.get((row['TIME OCC'], row['AREA']), 0), axis=1)
    CrimeSet['Average CClass By TA']=CrimeSet['Average CClass By TA'].round(decimals=3)

    # Make Freq F by Hour : felony ratio per hour
    # Order setted as ratio of felony
    order = [12, 0, 11, 10, 8, 6, 9, 13, 7, 14, 15, 16, 1, 18, 17, 20, 19, 21, 22, 2, 23, 5, 3, 4]
    CrimeSet['Freq F By Hour'] = CrimeSet['TIME OCC'].map(lambda x: order.index(x))

    # Make Freq F by AREA : felony ratio per area
    # Order setted as ratio of felony
    order = [1, 6, 7, 8, 21, 3, 17, 16, 15, 19, 9, 2, 14, 10, 11, 20, 18, 5, 13, 12, 4]
    CrimeSet['Freq F By AREA'] = CrimeSet['AREA'].map(lambda x: order.index(x))

    # Make Freq F by Victim Sex : felony ratio per sex of victim
    # Order setted as ratio of felony
    order = ['F','M','X']
    CrimeSet['Freq F By Sex'] = CrimeSet['Vict Sex'].map(lambda x: order.index(x))

    # Make Freq F by Descent : felony ratio per descent of victim
    # Order setted as ratio of felony
    order = ['Asian','B','W','Other']
    CrimeSet['Freq F By Descent'] = CrimeSet['Vict Descent'].map(lambda x: order.index(x))

    # Data of Age of victim into integer
    CrimeSet['Vict Age'] = CrimeSet['Vict Age'].astype(int)
    # Set age in unit of ten
    CrimeSet['Vict Age'] = CrimeSet['Vict Age'].apply(lambda x: math.floor(x/10)*10)
    # Clear -10
    CrimeSet = CrimeSet[CrimeSet['Vict Age'] != -10]
    CrimeSet = CrimeSet[CrimeSet['Vict Age'] != 0]

    # Make Freq F by Age : felony raio per Age of victim
    # Order setted as ratio of felony
    order = [10,70,60,50,80,90,40,30,20]
    CrimeSet['Freq F By Age'] = CrimeSet['Vict Age'].map(lambda x: order.index(x))
    

    # Incase of Mocodes = NAN, Ex Convict to DEFAULT(1)
    CrimeSet['Mocodes']=CrimeSet['Mocodes'].fillna(0)
    CrimeSet['Ex Convict'] = CrimeSet['Mocodes'].astype(str).str.split(' ').apply(lambda x: len(x))
    CrimeSet.drop('Mocodes',axis=1,inplace=True)

    # Weapon used ratio in grouped of Time and area
    grouped_data = CrimeSet.groupby(['TIME OCC', 'AREA'])
    mean_weapon = grouped_data['Weapon Or Not'].mean()
    CrimeSet['Average Weapon By TA'] = CrimeSet.apply(lambda row: mean_weapon.get((row['TIME OCC'], row['AREA']), 0), axis=1)
    CrimeSet['Average Weapon By TA']=CrimeSet['Average Weapon By TA'].round(decimals=3)

    # JA, JO, CC data are meaningless
    CrimeSet = CrimeSet[~CrimeSet['Status'].isin(['JO', 'JA','CC'])]

    # Make Freq F by Status : felony ratio per status of crime
    # Order setted as ratio of felony
    order = ['AO','IC','AA']
    CrimeSet['Freq F By Status'] = CrimeSet['Status'].map(lambda x: order.index(x))

    # Make Average Ex Convict By TA : Mean of Ex Convict in grouped of Time and area
    grouped_data = CrimeSet.groupby(['TIME OCC', 'AREA'])
    mean_convict = grouped_data['Ex Convict'].mean()
    CrimeSet['Average Ex Convict By TA'] = CrimeSet.apply(lambda row: mean_convict.get((row['TIME OCC'], row['AREA']), 0), axis=1)
    CrimeSet['Average Ex Convict By TA'] = CrimeSet['Average Ex Convict By TA'].round(decimals=3)

    # Get felony 2 data and count in grouped TIME OCC, AREA
    filtered_data = CrimeSet[CrimeSet['Crime Class'].isin([2])]
    count_filtered = filtered_data.groupby(['TIME OCC', 'AREA']).size().reset_index(name='Count_Filtered')
    # Group in TIME OCC, AREA and calculate occurances of count
    grouped_data = CrimeSet.groupby(['TIME OCC', 'AREA']).size().reset_index(name='Grouped_Count')

    # Make Felony Rate By TA (Target feature) : merge count_filtered and groiped_data based on TIME OCC and AREA
    merged_data = count_filtered.merge(grouped_data, on=['TIME OCC', 'AREA'], how='left')
    merged_data['Felony Rate By TA'] = merged_data['Count_Filtered'] / merged_data['Grouped_Count']

    # Merge merged_data into base data set and rename feature
    CrimeSet = CrimeSet.merge(merged_data[['TIME OCC', 'AREA', 'Felony Rate By TA']], on=['TIME OCC', 'AREA'], how='left')
    CrimeSet.rename(columns={'Felony Rate By TA_x': 'Felony Rate By TA'}, inplace=True)

    # Drop duplicate columns
    duplicate_columns = CrimeSet.columns[CrimeSet.columns.duplicated()]
    CrimeSet.drop(columns=duplicate_columns, inplace=True)

    # Drop used features
    drop_columns=['DATE OCC','AREA NAME','Crm Cd','Crm Cd Desc','Vict Sex','Vict Descent','Status']
    CrimeSet.drop(columns=drop_columns,axis=1,inplace=True)

    # Set data in type of float
    CrimeSet = CrimeSet.astype(float)
    
    print("Full Dataset Preprocessing Done!")
    return CrimeSet


def Scaling_Set(CrimeSet):
    # Scaling
    scaler = StandardScaler()
    crime_2022_c = CrimeSet.copy()

    # Set target and other features
    target = crime_2022_c['Felony Rate By TA']
    features = crime_2022_c.drop('Felony Rate By TA', axis=1)

    # Standard scaling features set and concat in crime_2022_scaled
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    crime_2022_scaled = pd.concat([features_scaled, target], axis=1)
    
    print("Scaling Done!")
    return crime_2022_scaled
