# Making the Census slightly easier to use

I'd venture a guess that half of all data science & analytics professionals have at some point interacted with Census data.

Recently, for an project studying how Covid-19 closures affected income & personal debt, I felt that the census data would be extremely valuable.

The last Canada census was taken in 2016, and the summarized census data can be downloaded from [the statscan website](https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/details/download-telecharger/comp/page_dl-tc.cfm?Lang=E) in a few different formats. The supported formats are CSV, TSV, XML & IVT. The cencus files can get pretty big. The largest CSV file was 1.6 Gigabytes (compressed). 

It took me (on not-the-best-internet) over an hour to download the file. There are other formats offered by statscan. The compressed tab-separted version has (unsurprisingly) the same size as the compressed CSV. The compressed XML version is over 2 Gigabytes in size. I'm really curious how many people downloaded the XML version. The unfamiliar format (for me) on the statscan website had the file extension `IVT`. After some initial googling, that turned out to be a proprietary format supported by a product called Beyond 20/20. Luckily, statscan provides a beyond 20/20 data browser on its website, which can be used to explore the dataset. **Disclaimer: I have not tried it because I'm much more comfortable scripting in python**

When working with large-ish data I default to using [apache parquet](https://parquet.apache.org/) for its excellent compression ratio and easy interop with pandas, so as a long term strategy it seemed reasonable to save the CSV data in parquet format.

Looking at the first few lines (on windows, so forgive the powershell command) gave me a sense of the column names & reasonable guesses for the types.

    > gc -TotalCount 10 98-401-X2016044_English_CSV_data.csv
    
```
"CENSUS_YEAR","GEO_CODE (POR)","GEO_LEVEL","GEO_NAME","GNR","GNR_LF","DATA_QUALITY_FLAG","CSD_TYPE_NAME","ALT_GEO_CODE","DIM: Profile of Dissemination Areas (2247)","Member ID: Profile of Dissemination Areas (2247)","Notes: Profile of Dissemination Areas (2247)","Dim: Sex (3): Member ID: [1]: Total - Sex","Dim: Sex (3): Member ID: [2]: Male","Dim: Sex (3): Member ID: [3]: Female"
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Population, 2016",1,1,35151728,...,...
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Population, 2011",2,2,33476688,...,...
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Population percentage change, 2011 to 2016",3,,5.0,...,...
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Total private dwellings",4,3,15412443,...,...
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Private dwellings occupied by usual residents",5,4,14072079,...,...
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Population density per square kilometre",6,,3.9,...,...
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Land area in square kilometres",7,,8965588.85,...,...
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","Total - Age groups and average age of the population - 100% data",8,5,35151730,17264200,17887530
2016,"01","0","Canada",4.0,5.1,"20000"," ","01","0 to 14 years",9,,5839570,2992925,2846645
```

Also, the statscan website has really good documentation on what each column represents. In this file, we have the census profile of `Canada, provinces, territories, census divisions (CDs), census subdivisions (CSDs) and dissemination areas (DAs)`. The column `GEO_CODE (POR)` has the geographic code for each region. The first few rows shows that there are some funny `...` values in the columns `Dim: Sex (3): Member ID: [2]: Male` and `Dim: Sex (3): Member ID: [3]: Female`. 

From the statscan website, I found that these codes have the following interpretations:

```
..
    not available for a specific reference period
...
    not applicable
E
    use with caution
F
    too unreliable to be published
r
    revised
x
    suppressed to meet the confidentiality requirements of the Statistics Act
```

I decided it would be worthwhile to clean up this column of type-mixed data before converting to parquet. Also, I found the original column names to be quite long, and unweildly to type for speed of though data exploration. So, I took it upon myself to rename the columns to be more friendly for pandas-oriented data analysis.

The original & new column names I settled on are:

| original name                                    | new name          | 
| ------------------------------------------------ | ----------------- |
| CENSUS_YEAR                                      | census_year       |
| GEO_CODE (POR)                                   | geo_code          |
| GEO_LEVEL                                        | geo_level         |
| GEO_NAME                                         | geo_name          |
| GNR                                              | gnr               |
| GNR_LF                                           | gnr_lf            |
| DATA_QUALITY_FLAG                                | data_quality_flag |
| CSD_TYPE_NAME                                    | csd_type_name     |
| ALT_GEO_CODE                                     | alt_geo_code      |
| DIM: Profile of Dissemination Areas (2247)       | profile_dim       |
| Member ID: Profile of Dissemination Areas (2247) | profile_member_id |
| Notes: Profile of Dissemination Areas (2247)     | notes             |
| Dim: Sex (3): Member ID: [1]: Total - Sex        | total             |
| Dim: Sex (3): Member ID: [2]: Male               | male              |
| Dim: Sex (3): Member ID: [3]: Female             | female            |

In order to read the origninal file efficiently into pandas, I supplied the dtypes and renamed the columns in the same operation. Of course I didn't do this the first few times I tried to read the file, and expectedly, pandas either took too long or crashed (depending on what else I was also doing on the computer) or complained afterwards when I was trying to manipulate the file that there wasn't enough memory. After reading, I set about cleanding the data & adding the 3 columns to store the missing data codes

    import numpy as np
    import pandas as pd
    import fastparquet as fp
    
    dtypes = {'census_year': 'category', 'geo_code': 'int', 
              'geo_level': 'category', 'geo_name': 'category', 
              'gnr': 'float', 'gnr_lf': 'float', 
              'data_quality_flag': 'category', 'csd_type_name': 'str', 
              'alt_geo_code': 'int', 'profile_dim': 'category', 
              'profile_member_id': 'category', 'notes': 'category', 
              'total': 'str', 'male': 'str', 'female': 'str'}
    
    names = list(dtypes.keys())
    
    df = pd.read_csv('98-401-X2016044_English_CSV_data.csv', 
                     header=0, 
                     names=names, 
                     dtype=dtypes) 
                     
    sym = ['...','..','E','F','x','r']   
    for c in ['total', 'male', 'female']:
       sym_col = f'{c}_sym'
       df[sym_col] = df[c]
       df.loc[~df[sym_col].isin(sym), sym_col] = np.nan
       df[sym_col] = df[sym_col].astype('category')
       df.loc[df[c].isin(sym), c] = np.nan
       df[c] = df[c].astype(float)

That's the extent of the modifications I made to the orignal dataset, before saving it as parquet. Here I've used fastparquet. But I bet pyarrow would get the job done equally as well. The thing I like about hive scheme is that it produces many smaller files instead of one large file, which I can exploit to upload the data to github with a little less pain in order to share with others. 

    fp.write('profile_cd_csd_da.parq', df, file_scheme='hive', compression='gzip')
    
I really hope this helps the next person that feels the need to consult the Canada Census Data. If you'd like to help, please add other datasets as parquet files to this repo.
