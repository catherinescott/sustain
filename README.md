# sustain
Scripts to run SuStaIn on PET data

Step 1: Merge relevent csv files
For SUVR data run: SUVR_data_merge.py
Takes csv fileson data path, will look for defined reference regions and PVC data. Will combine the data based on the data merge options for multi timepoint data
Output: csv file containing all the SUVR data ready for step 2

Step 2: Generate z-scores from csv files
For SUVR data run: gen_zscore_SUVR_modsel.py
ensure that the csv files to be used are all in datapath.
Will determine whether regions should be included and calculate the z-scores
Output: 2 csv files, one with zscores and another with z_max

Step 3: Run SuStaIn
For SUVR data run: run_sustain_GMM_SUVR_v1.py
Turn cross validation and remove zero subs off
Output: pickle files and plots

Step 4: Re-run SuStaIn and cross validation
For SUVR data run: run_sustain_GMM_SUVR_v1.py
Turn cross validation and remove zero subs on
Output: pickle files and plots

Step 5: getting the results
Use the results of the cross validation to determine the optimal number of subtypes and use this data.


