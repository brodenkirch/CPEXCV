import os
import sys
import pandas as pd
import numpy as np

def ms2kts(val):  #converts meters per second to knots
    return val * 1.94384449

def get_shear(ubot, utop, vbot, vtop):  #returns shear vector in degrees (directional) and knots (speed)
    #TOL = 1e-10           # Floating Point Tolerance
    ubot = ms2kts(ubot)
    utop = ms2kts(utop)
    vbot = ms2kts(vbot)
    vtop = ms2kts(vtop)
    shu = utop - ubot
    shv = vtop - vbot
    wdir = np.degrees(np.arctan2(-shu, -shv))
    if wdir < 0:
        wdir += 360
    #if np.fabs(wdir) < TOL:
        #wdir = 0.
    mag = np.sqrt(shu**2 + shv**2)
    
    return wdir, mag

sfc_heights = []
upper_lvl_cap_heights = []
heights_500m = []
deep_speed500m = []
deep_dir500m = []
dropsonde_filepath = os.path.join(os.getcwd(), 'Dropsonde_Metric_Calculations.csv')
dropsonde_df = pd.read_csv(dropsonde_filepath)

#loop through each dropsonde profile in the dropsonde metrics CSV and find its 'sfc' and upper level cap heights
for ip in range(len(dropsonde_df)):
    prof_day = str(dropsonde_df['Date'].iloc[ip])
    prof_time = str(dropsonde_df['Time'].iloc[ip])
    prof_sfc_pres = dropsonde_df['Sfc Pressure [mb]'].iloc[ip]
    prof_upper_cap_pres = dropsonde_df['Upper Level Cap [mb]'].iloc[ip]
    
    #make a datetime string in the format that's in the final dropsonde CSVs (YYYY-mm-dd HH:MM:SS) and locate that datetime's dropsonde data in the final_dropsonde CSV
    prof_datetime = prof_day[:4] + '-' + prof_day[4:6] + '-' + prof_day[6:8] + ' ' + prof_time[:2] + ':' + prof_time[2:4] + ':' + prof_time[4:6]
    drop_csv_path = os.path.join(os.getcwd(), prof_day, 'final_dropsonde_' + prof_day + '.csv')
    drop_day_df = pd.read_csv(drop_csv_path)
    df_at_time = drop_day_df[drop_day_df['Time [UTC]'] == prof_datetime]
    heights = df_at_time['Height [m]']
    pressures = df_at_time['Pressure [mb]']
    
    #find height corresponding to the 'Sfc Pressure [mb]' in the dropsonde metrics CSV
    sfc_height_index = abs(pressures - prof_sfc_pres).idxmin()
    sfc_height = heights.loc[sfc_height_index]
    sfc_heights.append(sfc_height)
    
    #find height corresponding to the 'Upper Level Cap [mb]' in the dropsonde metrics CSV
    cap_height_index = abs(pressures - prof_upper_cap_pres).idxmin()
    cap_height = heights.loc[cap_height_index]
    upper_lvl_cap_heights.append(cap_height)
    
    #calculate 500m - 7622.5m deep shear for each dropsonde and add this new field to the dropsonde metrics CSV
    ucomp = df_at_time['U Comp of Wind [m/s]']
    vcomp = df_at_time['V Comp of Wind [m/s]']
    
    height500_index = abs(heights - 500).idxmin()     #index is from the original drop_day_df (before filtering)
    height500 = heights.loc[height500_index]
    heights_500m.append(height500)
    
    ubot = ucomp.loc[height500_index]
    vbot = vcomp.loc[height500_index]
    
    utop = ucomp.loc[cap_height_index]
    vtop = vcomp.loc[cap_height_index]
    
    deep_dir, deep_speed = get_shear(ubot, utop, vbot, vtop)
    deep_dir = np.round(deep_dir, 1)
    deep_speed = np.round(deep_speed, 2)
    
    deep_speed500m.append(deep_speed)
    deep_dir500m.append(deep_dir)

#add the new sfc/upper level cap height fields to the dropsonde metrics CSV
dropsonde_df['Sfc Height [m]'] = sfc_heights  #will override the current 'Sfc Height [m]' column in the dropsonde metrics CSV
dropsonde_df['Upper Level Cap Height [m]'] = upper_lvl_cap_heights
dropsonde_df['500m Bottom Cap Profile Height [m]'] = heights_500m
dropsonde_df['500m Bottom Cap Deep Layer Speed Shear [kts]'] = deep_speed500m
dropsonde_df['500m Bottom Cap Deep Layer Directional Shear [deg]'] = deep_dir500m
final_name = os.path.join(os.getcwd(), 'Dropsonde_Metric_Calculations.csv')
dropsonde_df.to_csv(final_name, index = False)


