import numpy as np
import pandas as pd


icrg_data = pd.read_csv('ICRG/ICRGdataset.csv')
icrg_data = icrg_data[icrg_data['year'] > 2001]
icrg_data = icrg_data[icrg_data['year'] < 2020]

icrg_data['iyear_country'] = icrg_data['Country'] + icrg_data['year'].apply(str)
icrg_data.set_index('iyear_country')
wgi_data = pd.read_excel('WGI/wgidataset.xlsx', sheet_name=None)


column_target_name = ["Estimate", "StdErr", "NumSrc", "Rank", "Lower", "Upper"]
for i in wgi_data.keys():
    new_column_name = [i for i in range(len(wgi_data[i].columns) - 2)]
    for j in range(len(new_column_name)):
        new_column_name[j] = str(2002 + j // len(column_target_name)) + column_target_name[
            (j % len(column_target_name))]
    new_column_name = ['country', 'country_code'] + new_column_name
    wgi_data[i].columns = new_column_name

for i in wgi_data.keys():
    wgi_data[i] = wgi_data[i].set_index('country')
target = ['VoiceandAccountability', 'Political StabilityNoViolence', 'GovernmentEffectiveness', 'RegulatoryQuality',
          'RuleofLaw', 'ControlofCorruption']
for i in target:
    icrg_data[i] = np.full((len(icrg_data)), np.nan)


def wgi_2_icrg(iyear, country, target):


    if country not in target.index:
        return np.nan
    res = target.loc[country, str(iyear) + 'Estimate']
    return res


for i in target:

    for k, it in icrg_data.iterrows():
        icrg_data.loc[k, i] = wgi_2_icrg(it['year'], it['Country'], wgi_data[i])


gtd_target = ['count', 'vicinity', 'crit1', 'crit2', 'crit3', 'multiple', 'success', 'suicide', 'attacktype',
              'targettype', 'targetsubtype', 'natlty1', 'gname', 'guncertain1'
    , 'individual', 'claimed', 'weaptype', 'weapsubtype', 'nkill(mean)', 'nkill(max)', 'nwound(mean)', 'nwound(max)',
              'property', 'propextent(min)']
for i in gtd_target:
    icrg_data[i] = np.full((len(icrg_data)), np.nan)
gtd_data = pd.read_csv('gtd.csv')
gtd_data['country_year'] =   gtd_data['country_txt'] + gtd_data['iyear'].apply(lambda x:str(x-1))
group = gtd_data.groupby('country_year')
icrg_data = icrg_data.set_index("iyear_country")
for name, g in group:
    target_name = name[:-4] + str(int(name[-4:])+1)
    if name not in icrg_data.index.values.tolist():
        continue
    icrg_data.loc[name,'count'] = g.size # 总数
    icrg_data.loc[name,'vicinity'] = g['vicinity（mean)'].mean()
    icrg_data.loc[name,'crit1'] = g['crit1(mean)'].mean()
    icrg_data.loc[name,'crit2'] = g['crit1(mean)'].mean()
    icrg_data.loc[name,'crit3'] = g['crit1(mean)'].mean()
    icrg_data.loc[name,'multiple'] = g['multiple(mean)'].mean()
    icrg_data.loc[target_name,'success'] = g['success(mean)'].mean()
    icrg_data.loc[name,'suicide'] = g['suicide(mean)'].mean()
    icrg_data.loc[name,'attacktype'] = np.nan if g['attacktype1(z)'].mode().size==0 else g['attacktype1(z)'].mode()[0]
    icrg_data.loc[name,'targettype'] = np.nan if g['targtype1(z)'].mode().size==0 else g['targtype1(z)'].mode()[0]
    icrg_data.loc[name,'targetsubtype'] =  np.nan if g['targsubtype1(z)'].mode().size==0 else g['targsubtype1(z)'].mode()[0]
    icrg_data.loc[name,'natlty1'] = np.nan if g['natlty1(z)'].mode().size==0 else g['natlty1(z)'].mode()[0]
    icrg_data.loc[name,'gname'] =  np.nan if g['gname(z)'].mode().size==0 else g['gname(z)'].mode()[0]
    icrg_data.loc[name,'guncertain1'] = np.nan if g['guncertain1(z)'].mode().size==0 else g['guncertain1(z)'].mode()[0]
    icrg_data.loc[name,'individual'] = np.nan if g['individual(z)'].mode().size==0 else g['individual(z)'].mode()[0]
    icrg_data.loc[name,'claimed'] = np.nan if g['claimed(z)'].mode().size==0 else g['claimed(z)'].mode()[0]
    icrg_data.loc[name,'weaptype'] = np.nan if g['weaptype1(z)'].mode().size==0 else g['weaptype1(z)'].mode()[0]
    icrg_data.loc[name,'weapsubtype'] = np.nan if g['weapsubtype1(z)'].mode().size==0 else g['weapsubtype1(z)'].mode()[0]
    icrg_data.loc[name,'nkill(mean)'] = g['nkill(mean)(max)'].mean()
    icrg_data.loc[name,'nkill(max)'] = g['nkill(mean)(max)'].max()
    icrg_data.loc[name,'nwound(mean)'] = g['nwound(mean)(max)'].mean()
    icrg_data.loc[name,'nwound(max)'] = g['nwound(mean)(max)'].max()
    icrg_data.loc[name,'property'] = g['property(mean)'].mean()
    icrg_data.loc[name,'propextent(min)'] = g['propextent(min)'].min()

icrg_data.to_csv('data.csv')
