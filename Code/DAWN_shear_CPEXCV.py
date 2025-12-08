#DAWN Shear Calculations Using SHARPpy direct shear calculation method
#Results recorded in DAWN_Shear_calculations_CPEXCV.csv
import os
import sys
import pandas as pd
import numpy as np

use_normal_winds = True
use_median_winds = False   #determines if median shear is calculated or normal shear
use_average_winds = False  #determines if average shear is calculated or normal shear
sfc_height = 500   #the 500m cap eliminates noisy surface data, and all our current profiles have good data down to 500m (except 1 dropsonde) 
tolerance = 80    #DAWN profile has to have data within 80m of sfc_height and lowest_max_height

cases_dict = {'01': ['20220909', 'Eastern Atlantic', 'Isolated', '161000', '171000'], 
              '02': ['20220909', 'Eastern Atlantic', 'Isolated', '173500', '191000'], 
              '03': ['20220910', 'Eastern Atlantic', 'Isolated', '190500', '194000'], 
              '04': ['20220910', 'Eastern Atlantic', 'Isolated', '202500', '204500'], 
              '05': ['20220916', 'Eastern Atlantic', 'Isolated', '155500', '163500'], 
              '06': ['20220920', 'Eastern Atlantic', 'Isolated', '071000', '073500'], 
              '07': ['20220920', 'Eastern Atlantic', 'Isolated', '083000', '090000'], 
              '08': ['20220906', 'Eastern Atlantic', 'Organized', '110000', '120000'], 
              '09a': ['20220906', 'Eastern Atlantic', 'Organized', '133000', '143000'],
              '09b': ['20220906', 'Eastern Atlantic', 'Organized', '153000', '161000'],
              '10': ['20220906', 'Eastern Atlantic', 'Organized', '161000', '180000'], 
              '11': ['20220907', 'Eastern Atlantic', 'Organized', '130000', '134500'], 
              '12a': ['20220907', 'Eastern Atlantic', 'Organized', '134500', '151300'],
              '12b': ['20220907', 'Eastern Atlantic', 'Organized', '161800', '174500'],
              '13': ['20220907', 'Eastern Atlantic', 'Organized', '151300', '161800'],
              '14a': ['20220914', 'Eastern Atlantic', 'Organized', '101000', '115500'],
              '14b': ['20220914', 'Eastern Atlantic', 'Organized', '134000', '142800'],
              '15a': ['20220914', 'Eastern Atlantic', 'Organized', '115500', '131000'],
              '15b': ['20220914', 'Eastern Atlantic', 'Organized', '142800', '164500'],
              '16': ['20220916', 'Eastern Atlantic', 'Organized', '143000', '154000'],
              '17': ['20220916', 'Eastern Atlantic', 'Organized', '164000', '183500'],
              '18a': ['20220922', 'Eastern Atlantic', 'Organized', '054000', '061500'],
              '18b': ['20220922', 'Eastern Atlantic', 'Organized', '063500', '073600'],
              '18c': ['20220922', 'Eastern Atlantic', 'Organized', '080000', '083000'],
              '19': ['20220923', 'Eastern Atlantic', 'Organized', '092000', '143000'],
              '20': ['20220926', 'Eastern Atlantic', 'Organized', '072000', '111500'],
              '21': ['20220929', 'Eastern Atlantic', 'Organized', '103500', '134500'],
              '22': ['20220930', 'Eastern Atlantic', 'Organized', '134800', '143000'],
              '23': ['20220910', 'Eastern Atlantic', 'Scattered', '153500', '183000'],
              '24a': ['20220930', 'Eastern Atlantic', 'Scattered', '092000', '102000'],
              '24b': ['20220930', 'Eastern Atlantic', 'Scattered', '111000', '123000'],
              '24c': ['20220930', 'Eastern Atlantic', 'Scattered', '131800', '134000']}

def ms2kts(val):  #converts meters per second to knots
    return val * 1.94384449

def get_shear(ubot, utop, vbot, vtop):  #returns shear vector in degrees (directional) and knots (speed)
    #TOL = 1e-10           # Floating Point Tolerance
    ubot = ms2kts(ubot)
    utop = ms2kts(utop)
    vbot = ms2kts(vbot)
    vtop = ms2kts(vtop)
    
    bot_mag = np.sqrt(ubot**2 + vbot**2)
    bot_dir = np.degrees(np.arctan2(-ubot, -vbot))  #negating u and v gives the direction the wind is COMING from, not where its going (180 degree difference)
    if bot_dir < 0:
        bot_dir += 360
    top_mag = np.sqrt(utop**2 + vtop**2)    
    top_dir = np.degrees(np.arctan2(-utop, -vtop))  #negating u and v gives the direction the wind is COMING from, not where its going (180 degree difference)
    if top_dir < 0:
        top_dir += 360
    
    shu = utop - ubot
    shv = vtop - vbot
    shear_wdir = np.degrees(np.arctan2(-shu, -shv))
    if shear_wdir < 0:
        shear_wdir += 360
    #if np.fabs(shear_wdir) < TOL:
        #shear_wdir = 0.
    shear_mag = np.sqrt(shu**2 + shv**2)
    
    return shear_mag, shear_wdir, bot_mag, bot_dir, top_mag, top_dir


def get_median_shear(height, ucomp, vcomp, sfc_height, lowest_max_height, tolerance):  #returns shear vector in degrees (directional) and knots (speed) using median wind components at 500m and upper level cap

    #grab data at the height interval closest to the chosen 500m height (already checked if this height is within the tolerance)
    height500_index = abs(height - sfc_height).idxmin()   #index is from the new, re-ordered, re-indexed profile_use
    height500 = height[height500_index]
    ubot = ucomp[height500_index]
    vbot = vcomp[height500_index]
    
    #grab data one height interval below the chosen 500m height
    if height500_index != len(height) - 1:
        use_500_tolerance = tolerance * 1   #keeps the total tolerance range consistent
        height500_0 = height[height500_index + 1]
        ubot_0 = ucomp[height500_index + 1]
        vbot_0 = vcomp[height500_index + 1]
    else:  #if the chosen 500m height is the last index, then choose the data that is 2 points (height levels) above the chosen 500m height
        use_500_tolerance = tolerance * 2   #keeps the total tolerance range consistent
        height500_0 = height[height500_index - 2]
        ubot_0 = ucomp[height500_index - 2]
        vbot_0 = vcomp[height500_index - 2]    
    
    #grab data one height interval above the chosen 500m height index
    height500_2 = height[height500_index - 1]
    ubot_2 = ucomp[height500_index - 1]
    vbot_2 = vcomp[height500_index - 1]   

    #grab data at the height interval closest to the upper level cap (already checked if this height is within the tolerance)    
    top_height_index = abs(height - lowest_max_height).idxmin()  #index is from the new, re-ordered, re-indexed profile_use
    top_height = height[top_height_index]
    utop = ucomp[top_height_index]
    vtop = vcomp[top_height_index]
    
    #grab data one height interval below the chosen upper level cap
    top_height_0 = height[top_height_index + 1]
    utop_0 = ucomp[top_height_index + 1]
    vtop_0 = vcomp[top_height_index + 1]
    
    #grab data one height interval above the chosen upper level cap
    if top_height_index != 0:
        use_top_tolerance = tolerance * 1   #keeps the total tolerance range consistent
        top_height_2 = height[top_height_index - 1]
        utop_2 = ucomp[top_height_index - 1]
        vtop_2 = vcomp[top_height_index - 1]
    else:  #if the chosen upper level cap height is the first index (0), then choose the data that is 2 points (height levels) below the chosen upper level cap height
        use_top_tolerance = tolerance * 2   #keeps the total tolerance range consistent
        top_height_2 = height[top_height_index + 2]
        utop_2 = ucomp[top_height_index + 2]
        vtop_2 = vcomp[top_height_index + 2]

    bot_u_list = []
    bot_v_list = []
    top_u_list = []
    top_v_list = []
    
    #if the u,v values are within the use_500_tolerance/use_top_tolerance, then use them in the median calculation
    bot_u_list.append(ubot)  #already checked if this height is within the tolerance
    bot_v_list.append(vbot)  #already checked if this height is within the tolerance
    
    if abs(height500_2 - sfc_height) <= use_500_tolerance:
        bot_u_list.append(ubot_2)
        bot_v_list.append(vbot_2)
    if abs(height500_0 - sfc_height) <= use_500_tolerance:
        bot_u_list.append(ubot_0)
        bot_v_list.append(vbot_0)
        
    top_u_list.append(utop)  #already checked if this height is within the tolerance
    top_v_list.append(vtop)  #already checked if this height is within the tolerance
    
    if abs(top_height_2 - lowest_max_height) <= use_top_tolerance:
        top_u_list.append(utop_2)
        top_v_list.append(vtop_2)
    if abs(top_height_0 - lowest_max_height) <= use_top_tolerance:
        top_u_list.append(utop_0)
        top_v_list.append(vtop_0)
        
    #find the median bottom/top u,v wind components
    bot_u_median = np.median(bot_u_list)
    bot_v_median = np.median(bot_v_list)
    top_u_median = np.median(top_u_list)
    top_v_median = np.median(top_v_list)
    
    #calculate the median deep layer shear from the median u,v components at each level
    shear_mag, shear_wdir, bot_mag_median, bot_dir_median, top_mag_median, top_dir_median = get_shear(bot_u_median, top_u_median, bot_v_median, top_v_median)
    
    #record the heights used for the bottom level (500m) and the upper level cap, based on which height(s) had the median u,v values
    if bot_u_median == ubot:
        u_height500_use = height500
    elif bot_u_median == ubot_2:
        u_height500_use = height500_2
    elif bot_u_median == ubot_0:
        u_height500_use = height500_0
    elif bot_u_median == np.median([ubot, ubot_0]):  #the median value is calculated from only 2 points:  the ~500m point and one other point
        u_height500_use = (height500 + height500_0) / 2
    elif bot_u_median == np.median([ubot, ubot_2]):  #the median value is calculated from only 2 points:  the ~500m point and one other point
        u_height500_use = (height500 + height500_2) / 2
    #else:  #the median value is calculated from only 2 points:  the ~500m point and one other point
    
    if bot_v_median == vbot:
        v_height500_use = height500
    elif bot_v_median == vbot_2:
        v_height500_use = height500_2
    elif bot_v_median == vbot_0:
        v_height500_use = height500_0
    elif bot_v_median == np.median([vbot, vbot_0]):  #the median value is calculated from only 2 points:  the ~500m point and one other point
        v_height500_use = (height500 + height500_0) / 2
    elif bot_v_median == np.median([vbot, vbot_2]):  #the median value is calculated from only 2 points:  the ~500m point and one other point
        v_height500_use = (height500 + height500_2) / 2
    #else:  #the median value is calculated from only 2 points:  the ~500m point and one other point
    
    height500_use = (u_height500_use + v_height500_use) / 2
        
    if top_u_median == utop:
        u_top_height_use = top_height
    elif top_u_median == utop_2:
        u_top_height_use = top_height_2
    elif top_u_median == utop_0:
        u_top_height_use = top_height_0
    elif top_u_median == np.median([utop, utop_0]):  #the median value is calculated from only 2 points:  the top_height point and one other point
        u_top_height_use = (top_height + top_height_0) / 2
    elif top_u_median == np.median([utop, utop_2]):  #the median value is calculated from only 2 points:  the top_height point and one other point
        u_top_height_use = (top_height + top_height_2) / 2
    #else:  #the median value is calculated from only 2 points:  the top_height point and one other point
    
    if top_v_median == vtop:
        v_top_height_use = top_height
    elif top_v_median == vtop_2:
        v_top_height_use = top_height_2
    elif top_v_median == vtop_0:
        v_top_height_use = top_height_0
    elif top_v_median == np.median([vtop, vtop_0]):  #the median value is calculated from only 2 points:  the top_height point and one other point
        v_top_height_use = (top_height + top_height_0) / 2
    elif top_v_median == np.median([vtop, vtop_2]):  #the median value is calculated from only 2 points:  the top_height point and one other point
        v_top_height_use = (top_height + top_height_2) / 2
    #else:  #the median value is calculated from only 2 points:  the top_height point and one other point
    
    top_height_use = (u_top_height_use + v_top_height_use) / 2

    return shear_mag, shear_wdir, bot_mag_median, bot_dir_median, top_mag_median, top_dir_median, top_height_use, height500_use


def get_average_shear(height, ucomp, vcomp, sfc_height, lowest_max_height, tolerance):  #returns shear vector in degrees (directional) and knots (speed) using median wind components at 500m and upper level cap

    #grab data at the height interval closest to the chosen 500m height (already checked if this height is within the tolerance)
    height500_index = abs(height - sfc_height).idxmin()   #index is from the new, re-ordered, re-indexed profile_use
    height500 = height[height500_index]
    ubot = ucomp[height500_index]
    vbot = vcomp[height500_index]
    
    #grab data one height interval below the chosen 500m height
    if height500_index != len(height) - 1:
        use_500_tolerance = tolerance * 1   #keeps the total tolerance range consistent
        height500_0 = height[height500_index + 1]
        ubot_0 = ucomp[height500_index + 1]
        vbot_0 = vcomp[height500_index + 1]
    else:  #if the chosen 500m height is the last index, then choose the data that is 2 points (height levels) above the chosen 500m height
        use_500_tolerance = tolerance * 2   #keeps the total tolerance range consistent
        height500_0 = height[height500_index - 2]
        ubot_0 = ucomp[height500_index - 2]
        vbot_0 = vcomp[height500_index - 2]    
    
    #grab data one height interval above the chosen 500m height index
    height500_2 = height[height500_index - 1]
    ubot_2 = ucomp[height500_index - 1]
    vbot_2 = vcomp[height500_index - 1]   

    #grab data at the height interval closest to the upper level cap (already checked if this height is within the tolerance)    
    top_height_index = abs(height - lowest_max_height).idxmin()  #index is from the new, re-ordered, re-indexed profile_use
    top_height = height[top_height_index]
    utop = ucomp[top_height_index]
    vtop = vcomp[top_height_index]
    
    #grab data one height interval below the chosen upper level cap
    top_height_0 = height[top_height_index + 1]
    utop_0 = ucomp[top_height_index + 1]
    vtop_0 = vcomp[top_height_index + 1]
    
    #grab data one height interval above the chosen upper level cap
    if top_height_index != 0:
        use_top_tolerance = tolerance * 1   #keeps the total tolerance range consistent
        top_height_2 = height[top_height_index - 1]
        utop_2 = ucomp[top_height_index - 1]
        vtop_2 = vcomp[top_height_index - 1]
    else:  #if the chosen upper level cap height is the first index (0), then choose the data that is 2 points (height levels) below the chosen upper level cap height
        use_top_tolerance = tolerance * 2   #keeps the total tolerance range consistent
        top_height_2 = height[top_height_index + 2]
        utop_2 = ucomp[top_height_index + 2]
        vtop_2 = vcomp[top_height_index + 2]

    bot_u_list = []
    bot_v_list = []
    top_u_list = []
    top_v_list = []
    
    #if the u,v values are within the use_500_tolerance/use_top_tolerance, then use them in the mean calculation
    bot_u_list.append(ubot)  #already checked if this height is within the tolerance
    bot_v_list.append(vbot)  #already checked if this height is within the tolerance
    
    if abs(height500_2 - sfc_height) <= use_500_tolerance:
        bot_u_list.append(ubot_2)
        bot_v_list.append(vbot_2)
    if abs(height500_0 - sfc_height) <= use_500_tolerance:
        bot_u_list.append(ubot_0)
        bot_v_list.append(vbot_0)
        
    top_u_list.append(utop)  #already checked if this height is within the tolerance
    top_v_list.append(vtop)  #already checked if this height is within the tolerance
    
    if abs(top_height_2 - lowest_max_height) <= use_top_tolerance:
        top_u_list.append(utop_2)
        top_v_list.append(vtop_2)
    if abs(top_height_0 - lowest_max_height) <= use_top_tolerance:
        top_u_list.append(utop_0)
        top_v_list.append(vtop_0)
        
    #find the mean bottom/top u,v wind components
    bot_u_mean = np.mean(bot_u_list)
    bot_v_mean = np.mean(bot_v_list)
    top_u_mean = np.mean(top_u_list)
    top_v_mean = np.mean(top_v_list)
    
    #calculate the mean deep layer shear from the mean u,v components at each level
    shear_mag, shear_wdir, bot_mag_mean, bot_dir_mean, top_mag_mean, top_dir_mean = get_shear(bot_u_mean, top_u_mean, bot_v_mean, top_v_mean)
    
    #calculate the heights used for the bottom level (500m) and the upper level cap (mean of the 3 heights you used)
    height500_use = (height500 + height500_0 + height500_2) / 3
    top_height_use = (top_height + top_height_0 + top_height_2) / 3

    return shear_mag, shear_wdir, bot_mag_mean, bot_dir_mean, top_mag_mean, top_dir_mean, top_height_use, height500_use


# convective_type_dict = {'Isolated': ['20170610', '20170624', '20210822'], 
#                         'Organized': ['20170615', '20170616', '20170615', '20170601', '20170606', '20170617', '20170619', '20170620', '20210821', '20210824', '20210828'], 
#                         'Scattered': ['20170602', '20170611']}

dropsonde_filepath = os.path.join(os.getcwd(), 'Dropsonde_Metric_Calculations_CPEXCV.csv')
dropsonde_df = pd.read_csv(dropsonde_filepath)

#MAKE SURE THAT lowest_max_height = lowmax_hght (currently 7622.5) FROM download_files.py
#lowest_max_height = dropsonde_df['Max Profile Height [m]'].sort_values(ascending = True).iloc[0]  #dropsonde lowest max height (7622.5)
lowest_max_height = 7622.5
#print ('Do you want to change the lowest_max_height???')
assert lowest_max_height == 7622.5, "Lowest max height does is not 7622.5 anymore"

#CODE FOR 500m - 7622.5m DEEP SHEAR

shear_dict = {'Date': [], 'Time': [], 'Case': [], 'Primary Convective Type': [], 'Region': [], 
                    '500m Bottom Cap Deep Layer Speed Shear [kts]': [], 
                    '500m Bottom Cap Deep Layer Directional Shear [deg]': [],
                    '500m Wind Speed [kts]': [], '500m Wind Direction [deg]': [],
                    'Upper Level Cap Wind Speed [kts]': [], 'Upper Level Cap Wind Direction [deg]': [],
                    'DAWN PBL Speed Shear [kts]': [], 'DAWN PBL Directional Shear [deg]': [],
                    'DAWN Mid Layer Speed Shear [kts]': [], 'DAWN Mid Layer Directional Shear [deg]': [],
                    'DAWN Upper Layer Speed Shear [kts]': [], 'DAWN Upper Layer Directional Shear [deg]': [],
                    '500m Bottom Cap Profile Height [m]': [], 'Upper Level Cap Height [m]': [], 
                    'Environment Falling In': [], 'Environment Falling In Ambiguous': [], 'Convective Lifecycle': []}
                    ###^^^ last 3 fields are dummy variables put in so that dropsonde and DAWN shear can 
                    ###    be plotted simultaneously easier (see Deep_Shear_Scatter.py)

#get all unique dates from dropsonde_df and converts them all to strings
days = dropsonde_df.Date.unique().astype(str)

for day in days:  #calculate DAWN shear for each qualifying DAWN profile for each day
    dawn_csv_path = os.path.join(os.getcwd(), day, 'final_dawn_' + day + '.csv')
    dawn_df = pd.read_csv(dawn_csv_path)
    dawn_group = dawn_df.groupby(['Time [UTC]'])
    
    for profile in dawn_group:  #calculate DAWN shear for each qualifying DAWN profile
        time = ''.join(profile[0].split(':'))[-6:]  #just grabs the time from the date/time string in format HHMMSS
        
        #order the profile based on the height values (highest to lowest) and reset the indices
        profile_use = profile[1].sort_values(by = ['Height [m]'], ascending = False, ignore_index = True)
        height = profile_use['Height [m]']
        
        height500_index = abs(height - sfc_height).idxmin()    #index is from the new, re-ordered, re-indexed profile_use
        height500 = height[height500_index]

        top_height_index = abs(height - lowest_max_height).idxmin()    #index is from the new, re-ordered, re-indexed profile_use
        top_height = height[top_height_index]
        
        #if there is enough near-surface data (# data points below 1km); NOTE: data filter below 1km should never exceed 30 points (see 12/13/2021 Action Items in your Box notes)
        #if there is data close enough (tolerance) to lowest_max_height and sfc_height
        if (len(height[height < 1000]) >= 20) and (abs(top_height - lowest_max_height) <= tolerance) and (abs(height500 - sfc_height) <= tolerance):
            ucomp = profile_use['U Comp of Wind [m/s]']
            vcomp = profile_use['V Comp of Wind [m/s]']

            if use_median_winds:  #use median u,v winds for the wind shear calculations
                deep_speed, deep_dir, bot_mag, bot_dir, top_mag, top_dir, top_height, height500 = get_median_shear(height, ucomp, vcomp, sfc_height, lowest_max_height, tolerance)
                
            elif use_average_winds:  #use mean u,v winds for the wind shear calculations
                deep_speed, deep_dir, bot_mag, bot_dir, top_mag, top_dir, top_height, height500 = get_average_shear(height, ucomp, vcomp, sfc_height, lowest_max_height, tolerance)
                
            else: #use the u,v winds nearest the 500m and upper level cap heights
                ubot = ucomp[height500_index]
                vbot = vcomp[height500_index]
                
                #grab data at the height interval closest to the upper level cap
                utop = ucomp[top_height_index]
                vtop = vcomp[top_height_index]
                
                deep_speed, deep_dir, bot_mag, bot_dir, top_mag, top_dir = get_shear(ubot, utop, vbot, vtop)
            
            deep_speed = np.round(deep_speed, 2)
            deep_dir = np.round(deep_dir, 2)
            bot_mag = np.round(bot_mag, 2)
            bot_dir = np.round(bot_dir, 2)
            top_mag = np.round(top_mag, 2)
            top_dir = np.round(top_dir, 2)
            
            #find the case, convective type, and region for the given DAWN profile
            convective_type = '--'
            case = '--'
            region = '--'
            for case_num in cases_dict:
                full_date = cases_dict[case_num][0]
                if day == full_date:
                    if len(cases_dict[case_num]) == 3:  #if full_date does not have multiple cases and the time range of the 1 case is one CONTINUOUS range
                        case = int(case_num[0:2])       #works as long as you don't have > 99 cases
                        region = cases_dict[case_num][1]
                        convective_type = cases_dict[case_num][2]
                        break
                    else:  #if full_date has multiple cases or the time range of the one case is not continuous
                        if time >= cases_dict[case_num][3] and time < cases_dict[case_num][4]:  #only works if the case's start/end times are on the same UTC day! 
                            case = int(case_num[0:2])
                            region = cases_dict[case_num][1]
                            convective_type = cases_dict[case_num][2]
                            break
                        else:
                            pass
            
            if case == '--':  #if the given DAWN profile is not within any case's time range(s), don't add that DAWN profile to the CSV
                continue
            
            #add the DAWN profile's respective data to the shear dictionary (and eventual DAWN shear CSV)
            shear_dict['Date'].append(day)
            shear_dict['Time'].append(time)
            shear_dict['Case'].append(case)
            shear_dict['Primary Convective Type'].append(convective_type)
            shear_dict['Region'].append(region)
            shear_dict['500m Bottom Cap Profile Height [m]'].append(height500)
            shear_dict['Upper Level Cap Height [m]'].append(top_height)
            shear_dict['500m Bottom Cap Deep Layer Speed Shear [kts]'].append(deep_speed)
            shear_dict['500m Bottom Cap Deep Layer Directional Shear [deg]'].append(deep_dir)
            shear_dict['500m Wind Speed [kts]'].append(bot_mag)
            shear_dict['500m Wind Direction [deg]'].append(bot_dir)
            shear_dict['Upper Level Cap Wind Speed [kts]'].append(top_mag)
            shear_dict['Upper Level Cap Wind Direction [deg]'].append(top_dir)
            shear_dict['DAWN PBL Speed Shear [kts]'].append('--')
            shear_dict['DAWN PBL Directional Shear [deg]'].append('--')
            shear_dict['DAWN Mid Layer Speed Shear [kts]'].append('--')
            shear_dict['DAWN Mid Layer Directional Shear [deg]'].append('--')
            shear_dict['DAWN Upper Layer Speed Shear [kts]'].append('--')
            shear_dict['DAWN Upper Layer Directional Shear [deg]'].append('--')
            shear_dict['Environment Falling In'].append('Clear Near')   #dummy variable, don't take literally
            shear_dict['Environment Falling In Ambiguous'].append('--')
            shear_dict['Convective Lifecycle'].append('--')

#convert the DAWN shear dict to the DAWN shear DataFrame to the DAWN shear CSV
if use_median_winds:
    final_name = os.path.join(os.getcwd(), 'DAWN_Shear_Calculations_medianShear_CPEXCV.csv')
if use_average_winds:
    final_name = os.path.join(os.getcwd(), 'DAWN_Shear_Calculations_averageShear_CPEXCV.csv')
if use_normal_winds:
    final_name = os.path.join(os.getcwd(), 'DAWN_Shear_Calculations_CPEXCV.csv')

DAWN_shear_df = pd.DataFrame.from_dict(shear_dict)  
DAWN_shear_df.to_csv(final_name, index = False)


     