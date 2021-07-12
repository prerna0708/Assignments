import world_bank_data as wb
from meteostat import Point, Daily, Stations
from datetime import datetime,date
from os import path
import numpy as np
import pandas as pd

def fetch_country_daily_temperature(country, start_date, end_date, num_of_stations):

    stations = list(
        Stations().region(country).fetch(num_of_stations).index
        )
    if '02956' in stations:
        stations.remove('02956')
    if '68040' in stations:
        stations.remove('68040')
    if '68240' in stations:
        stations.remove('68240')
    if '68038' in stations:
        stations.remove('68038')
    if 'ZMUB0' in stations:
        stations.remove('ZMUB0')
    if '91737' in stations:
        stations.remove('91737')
    if 'KQRC0' in stations:
        stations.remove('KQRC0')

    df_temp_data = Daily(stations, start_date, end_date).fetch()

    df_temp_data['country'] = country
    return df_temp_data[['country','tmax','tmin','tavg','pres']]

def fetch_countries_daily_temperature(start_date, end_date):

    df_country_info = pd.read_csv(
        'data/dim_all_country_static_info.csv',
        index_col = 'id')

    df_temp_data = pd.DataFrame(
        columns = ['country','tmax','tmin','tavg','pres']
        )
    df_temp_data.index.names = ['date']

    for con in df_country_info.index:

        df_temp_data = df_temp_data.append(
            fetch_country_daily_temperature(
                df_country_info.loc[con]['iso2Code'],
                start_date, end_date,
                df_country_info.loc[con]['num_of_weather_station']
            )
        )
        print('finish downloading country:%s'%con)

    df_temp_data.index = pd.MultiIndex.from_arrays(
        [
            df_temp_data.index.map(
                lambda r : r[0] if type(r) == tuple else None
                ),

            df_temp_data.index.map(
                lambda r : r[1] if type(r) == tuple else r
                )
        ],
        names = ['station_id', 'date']
    )

    return df_temp_data

def aggregate_daaily_temp_from_weather_station(df_temp_data):

    df_temp_day_coun['date'] = pd.to_datetime(df_temp_day_coun.date)
    df_temp_day_coun = df_temp_day_coun.set_index(['country','year','date'])

    df_temp_data['date'] = pd.to_datetime(df_temp_data.date)

    df_temp_data['year'] = df_temp_data.date.dt.year

    df_temp_data['tmax'] = df_temp_data.tmax.apply(lambda r: r if r <= 50 else None) \
        .fillna(method = 'ffill')

    df_temp_data['tavg'] = df_temp_data.tavg.apply(lambda r: r if r <= 50 else None) \
        .fillna(method = 'ffill')

    df_temp_day_coun = df_temp_data.groupby(['country', 'year','date']).agg({
        'tmax': 'max',
        'tmin': 'min',
        'tavg': 'mean',
        'pres': 'max'
    })

    return df_temp_day_coun

def generate_heat_wave_by_temp(start_date, end_date):

    raw_weather_station_file = 'data/dim_temp_data_day_%d_%d.csv'%(start_date.year, end_date.year)

    daily_country_weather_file = 'data/dim_temp_data_%d_%d.csv'%(start_date.year, end_date.year)

    heat_wave_gen_file = 'data/dim_temp_gen_heat_wave_%d_%d.csv'%(start_date.year, end_date.year)

    if ~ path.exists(raw_weather_station_file):

        df_temp_data = fetch_countries_daily_temperature(start_date, end_date)

        df_temp_data.to_csv(raw_weather_station_file)

    else:

        df_temp_data = pd.read_csv(raw_weather_station_file)

    if ~ path.exists(daily_country_weather_file):

        df_temp_day_coun = aggregate_daaily_temp_from_weather_station(df_temp_data)

        df_temp_day_coun.to_csv(daily_country_weather_file)

    else:
        df_temp_day_coun = pd.read_csv(daily_country_weather_file)

    ## calculate the 85 percentile threshold

    df_temp_day_coun_ma15 = \
        df_temp_day_coun.groupby('country')[['tmax','tmin']].rolling(15).mean().droplevel(0)

    df_temp_threshold = df_temp_day_coun_ma15.groupby(
        ['country', 'year']
    ).quantile(0.85).rename(
        columns = {'tmin': 'tmin_thres', 'tmax': 'tmax_thres'}
        )

    df_temp_threshold['tmax_thres'] = df_temp_threshold.apply(
        lambda r: 32 if r['tmax_thres'] <= 32 else r['tmax_thres'], axis =1
    )

    df_temp_day_coun_thres = pd.merge(
        df_temp_day_coun, df_temp_threshold,
        left_index = True,
        right_index = True
    )
    ## calculate EHF (Excess Heat Factor)

    avg_last_3 = df_temp_day_coun['tavg'].copy()

    for i in range(1, 3):
        avg_last_3 += df_temp_day_coun.groupby(['country'])['tavg'].shift(-i)

    avg_last_3 = (avg_last_3/3).fillna(method = 'ffill')

    avg_3_32 = df_temp_day_coun.groupby(['country'])['tavg'].shift(-3).copy()

    for i in range(4, 33):
        avg_3_32 += df_temp_day_coun.groupby(['country'])['tavg'].shift(-i)

    avg_3_32 = (avg_3_32/30).fillna(method = 'ffill')

    EHI_accl = avg_last_3 - avg_3_32

    df_EHI_sig = pd.merge(
        avg_last_3,
        df_temp_day_coun.groupby(['country','year'])['tavg'].quantile(0.95).rename('tavg95'),
        left_index = True,
        right_index =True
    )

    EHI_sig = avg_last_3 - df_EHI_sig.tavg95

    df_EHF = pd.merge(
        EHI_accl.rename('EHI_accl'),
        EHI_sig.rename('EHI_sig'),
        left_index = True,
        right_index = True
    )

    df_EHF['EHI_accl'] = df_EHF.EHI_accl.apply(lambda r: r if r > 1 else 1)


    # three threshold to filter out the heat wave events: is_hw_ehf, is_hw_ctx85, is_hw_ctn85
    df_temp_day_coun['is_hw_ehf'] = df_EHF['EHI_accl'] * df_EHF['EHI_sig'] > 0

    df_temp_day_coun['is_hw_ctx85'] = \
        df_temp_day_coun.tmax > (df_temp_day_coun_thres.tmax_thres +0.1)

    df_temp_day_coun['is_hw_ctn85'] = \
        df_temp_day_coun.tmin > (df_temp_day_coun_thres.tmin_thres + 0.1)

    df_thres_filter = df_temp_day_coun[
        df_temp_day_coun.is_hw_ctx85 & df_temp_day_coun.is_hw_ctn85
        & df_temp_day_coun.is_hw_ehf
        ].reset_index(level = 2)[['date','tmax']]

    # filter out the days with duration longer than three days.
    df_thres_filter['last_date'] = \
        df_thres_filter.groupby(['country', 'year'])['date'].shift(1)
    df_thres_filter['next_date'] = \
        df_thres_filter.groupby(['country', 'year'])['date'].shift(-1)

    df_thres_filter['start_flag'] = df_thres_filter.apply(
        lambda r: (r['date'] - r['last_date']).days > 1
        , axis = 1)
    df_thres_filter['end_flag'] = df_thres_filter.apply(
        lambda r: (r['next_date'] - r['date']).days > 1
        , axis = 1)

    df_thres_filter = df_thres_filter[
        (df_thres_filter.start_flag & ~df_thres_filter.end_flag) |
        (~df_thres_filter.start_flag & df_thres_filter.end_flag) &
        ~df_thres_filter.last_date.isna()
    ]

    df_thres_filter['end_date'] = df_thres_filter.groupby(['country','year'])['date'].shift(-1)
    df_thres_filter = df_thres_filter[df_thres_filter.start_flag]
    df_thres_filter['duration'] = df_thres_filter.apply(lambda r: (r['end_date'] - r['date']).days, axis=1)
    df_thres_filter = df_thres_filter[df_thres_filter.duration >=3]

    df_thres_filter = \
        df_thres_filter.drop(columns = ['start_flag','end_flag', 'last_date', 'next_date'])

    df_thres_filter['country_code'] = df_thres_filter.index.get_level_values(0)

    df_temp_day_coun_ = df_temp_day_coun.reset_index(level =2).copy()

    df_thres_filter[['avg_maximum_temp', 'maximum_temp']] = df_thres_filter.apply(
        lambda r :
            df_temp_day_coun_[
                df_temp_day_coun_.date.between(r['date'], r['end_date'])
                ].loc[r['country_code']]['tmax'].agg(['mean', 'max']), axis =1
                ).rename(columns = {'mean':'avg_maximum_temp', 'max': 'maximum_temp'}
                )

    df_hw_gen = df_thres_filter.groupby(['country', 'year']).agg(
        {
            'duration' : ['sum', 'max'],
            'date': 'nunique',
            'maximum_temp': 'max',
            'avg_maximum_temp': 'mean'
        }
    )
    df_hw_gen.columns = ['HWF','HWD','HWN','HWA','HWM']

    return df_hw_gen

class WBDIndicatorFetcher(object):

    def __init__(self):
        pass

    def construct_static_info(self):

        self.df_country_info = wb.get_countries()
        self.df_country_info = self.df_country_info[self.df_country_info.region !='Aggregates']

        from meteostat import Stations

        ## fetch the number of weather stations for each country.
        self.df_country_info['num_of_weather_station'] = self.df_country_info.apply(
            lambda r: Stations().region(r['iso2Code']).count() ,axis=1
            )
        ## fetch the land area for each country.
        self.fetch_countries_indicator(
                indicator = 'AG.LND.TOTL.K2',
                col_name = 'land_area_sq_km',
                spec_year = '2018'
                )

        self.country_codes = self.df_country_info.index
        self.df_country_info.index.names = ['country']


    def construct_panel_data(self, start_year = 1980, end_year = 2020):

        ## construct panel data
        self.start_year = start_year
        self.end_year = end_year

        self.df_country_ts = pd.DataFrame(
            index = pd.MultiIndex.from_product(
                [
                    self.country_codes ,
                    pd.date_range(start = str(start_year), end = str(end_year),freq ='Y').to_period('Y').astype(str)
                ]
            )
        )
        self.df_country_ts.index.names = ['country', 'year']


    def fetch_countries_indicator_ts(
        self, indicator, col_name):
        '''
            add one more indicator to the df_country_ts
        '''

        for country in self.country_codes:
            df_country_ts.loc[country, col_name] = wb.get_series(
                indicator, country = country, simplify_index = True
                ).loc[self.start_year:self.end_year].values


    def fetch_countries_indicator(
        self, indicator, col_name, spec_year):

        self.df_country_info[col_name] = wb.get_series(
                indicator, date = spec_year, simplify_index = True,
                id_or_value = 'id'
                )


    def fetch_countries_indicators(self, indicator_maps):

        for indicator, col_name in indicator_maps.items():

            self.df_country_ts[col_name] = wb.get_series(
                        indicator, simplify_index = True, id_or_value = 'id'
                    ).reset_index(level =0).loc[np.arange(self.start_year, self.end_year).astype(str)] \
                    .set_index('Country',append =True).reorder_levels([1, 0], axis=0).sort_index()

def build_feature_data():

    start_year, end_year = 2020, 2020
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 1)

    feature_file = 'data/dim_all_country_info_%d_%d.csv'%(start_year, end_year)
    daily_country_weather_file = 'data/dim_temp_data_%d_%d.csv'%(start_year, end_year)

    # data from world bank
    indicator_maps = {
        'SP.POP.TOTL' : 'total_population',
        'SP.URB.TOTL.IN.ZS' : 'urban_pop_ratio',
        'AG.LND.FRST.ZS' : 'forest_area_ratio',
        'NY.GDP.MKTP.KD.ZG': 'gdp_growth_rate',
        'NY.GDP.MKTP.CD': 'gdp_growth_usd',
        'EN.ATM.CO2E.KT': 'co2_emission_kt',
        'AG.LND.AGRI.ZS': 'agri_land_ratio',
        'EN.ATM.METH.KT.CE': 'methane_emission_kt',
        'AG.PRD.LVSK.XD': 'livestock_prod_ind',
        'AG.PRD.FOOD.XD': 'food_prod_ind'
    }


    wbd = WBDIndicatorFetcher()
    wbd.construct_static_info()
    wbd.construct_panel_data()
    wbd.fetch_countries_indicators(indicator_maps)

    df_country_ts = pd.merge(
        wbd.df_country_ts,
        wbd.df_country_info,
        left_index= True,
        right_index =True
    ).reset_index().set_index(['iso2Code', 'year'])

    df_country_ts.index.names = ['country', 'year']

    df_country_ts.index.set_levels(
         np.arange(start_year,end_year),1,
         inplace= True
    )

    # data from meteostat
    if ~path.exists():
        df_hw_gen = generate_heat_wave_by_temp(start_date, end_date)
    else:
        df_hw_gen = pd.read_csv(daily_country_weather_file)

    df_country_ts_dim = pd.merge(
        df_country_ts,
        df_hw_gen,
        left_index= True,
        right_index =True,
        how = 'left'
    )

    df_country_ts_dim = pd.read_csv(daily_country_weather_file)

    df_country_ts_dim[['tmp_mean', 'tmp_median']] = df_temp_day_coun.groupby(
            ['country', 'year']
        )['tavg'].agg(['mean', 'median']).rename(
            columns = {'mean':'temp_mean', 'median': 'temp_median'}
            )

    df_country_ts_dim.to_csv(feature_file)


    # start_date = datetime(2019, 1, 1)
    # end_date = datetime(2019, 12, 1)
    #
    # df_hw_gen = generate_heat_wave_by_temp(df_temp_day_coun, start_date, end_date)
    #
    # df_hw_gen.to_csv(heat_wave_gen_file)
