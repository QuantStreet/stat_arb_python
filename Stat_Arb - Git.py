# -*- coding: utf-8 -*-
"""

@author: DF

"""

"""
The purpose of the script is to showcase ability (it is not an alpha project):
    
Cleans the data
Assumes Alpha to be ebitda over 12 months
Uses quadratic programming to optimize portfolio selection under trading 
constraints 
Loops over pre determined rebalance points and conditions to create a 
realistic backtest

Although it is not an alpha project, designing and testing alpha in this 
script is very straightforward. It is also adaptable to be functions or classes

The motivation behind this is that managing a portfolio can be quite different
than doing alpha research and this script is designed to account for the 
differences

NB: The script development, data and intellectual property does NOT overlap 
with my paid career history

Data Source: Quandl/Sharadar


Occasionally there is a SettingWithCopyWarning from eval(), but it's not valid

This script is 3 different pieces put together in one script for presentation 
purposes

Ignore the time portion of datetime64, I'm still considering whether it can 
be useful for anything
"""



import os
import sys as sys
import warnings as warnings

import numpy as np
import pandas as pd

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import cvxopt as cvxopt

from datetime import timedelta
import sklearn as sk
import sklearn.preprocessing as pre
from functools import reduce
import yfinance as yf

%matplotlib qt5



# module control:
if sys.version_info[:2] != (3, 8): warnings.warn('Built using Python 3.8')

modules_dict = {'np':'1.19.2', 'pd':'1.3.2', 'matplotlib':'3.4.3', 
                 'cvxopt':'1.2.0', 'sk':'0.24.2', 'yf':'0.1.63'}
for pack, vrs in modules_dict.items(): 
   if(eval(pack + '.__version__') != vrs):
       warnings.warn('Built using ' + pack + ' ' + vrs)
       
del matplotlib
del sk

# set up data directory containing Sharadar files

dataWD = 'data path...'

SEP = pd.read_csv(dataWD + 'SHARADAR_SEP.csv')
SEP['date']=pd.to_datetime(SEP['date'])
SEP.rename(columns = {'date':'datadate'}, copy = 0, inplace = 1)
SEP = SEP.sort_values(['ticker', 'datadate'])

SEP = SEP.loc[SEP.datadate.dt.year < 2016, :] # debug/testing

# To reduce the amount of processing time while keeping bias to none
# remove all tickers that never breach minimum price becuase they don't make
# any difference in the analysis
rem_low_price = (
    SEP
    .groupby('ticker')
    .closeunadj
    .agg(max)
    .loc[lambda x: x < 6].index
    )

SEP = SEP.query("ticker not in @rem_low_price")

SF1 = pd.read_csv(dataWD + 'SHARADAR_SF1.csv')
SF1['datekey']=pd.to_datetime(SF1['datekey'])
SF1 = SF1.loc[SF1.datekey <= SEP.datadate.max()]

tickers = pd.read_csv(dataWD + 'SHARADAR_TICKERS.csv', low_memory = False)
tickers.replace('None', np.nan, inplace = True)
tickers.set_index('ticker', inplace = True)

my_tic = SEP['ticker'].unique()

assert not np.any(SEP.ticker.isna().values)

pd_day_all = np.union1d(SEP.datadate.unique(), SF1.datekey.unique())

pd_day_all = pd.DataFrame(pd_day_all); my_tic=pd.DataFrame(my_tic)

pd_day_all.rename(columns = {0:'datadate'}, copy = 0, inplace = 1)
my_tic.rename(columns = {0:'ticker'}, copy = 0, inplace = 1)

pd_day_all['key']=1; my_tic['key']=1

day_ticker_all = pd_day_all.merge(
    my_tic, 
    how = 'outer', 
    on = 'key', 
    copy=False)

min_max_dates = (
    SEP
    .query("close == close")
    .groupby('ticker')
    .agg({'datadate' : ['min', 'max']})
    )

day_ticker_all = day_ticker_all.merge(
    min_max_dates.droplevel(0, axis = 1), 
    how = 'left', 
    left_on = 'ticker', 
    right_index = True, 
    copy = False
    )

day_ticker_all = day_ticker_all.query("datadate>=min & datadate<=max")

SEP = SEP.merge(
    day_ticker_all[['datadate', 'ticker']], 
    how = 'outer', 
    on = ['datadate', 'ticker'], 
    copy = False)

SEP.sort_values(['ticker', 'datadate'], inplace = True)

SEP['dividends'].fillna(0, inplace = True)

# The close feature is not adjusted to include dividends
SEP['ret'] = (
    SEP
    .query('close == close')
    .groupby('ticker', group_keys = False)
    .apply(lambda x: (x.close + x.dividends) / x.close.shift() - 1)
    )

tickers.loc[:, ['sector', 'industry']] = (
    tickers
    .query('siccode == siccode and table == "SEP"')
    .groupby(
        tickers
        .query('siccode == siccode and table == "SEP"')
        .siccode
        .astype('str')
        .str[:3])
    [['sector', 'industry']]
    .apply(lambda x: x.fillna(method = 'ffill').fillna(method = 'bfill'))
    )

merge_tickers = tickers.query('table == "SEP"')[
        ['exchange', 'sector', 'category', 
         'famaindustry', 'siccode', 'industry', 'name']
    ]

merge_tickers_drop = merge_tickers.query(
    'industry == "" \
    or sector != sector \
    and siccode == siccode \
    and category == "Domestic" \
    and siccode != 9995')
        
# Drop missing labels, assume that they are small names that would have
# gotten filtered out in either case
if merge_tickers_drop.shape[0] < 100:
    merge_tickers = merge_tickers.query(
        'ticker not in @merge_tickers_drop.index')
else: raise ValueError('There are too many empty classifications!')

SEP = SEP.merge(
    merge_tickers, 
    left_on = 'ticker', 
    right_index = True, 
    how = 'inner', 
    copy = False)

SEP = SEP.query("category == 'Domestic'")

assert(all((
    SEP
    .loc[SEP.datadate.isin(SEP.datadate.unique()[::120])]
    .groupby('datadate')
    .ticker
    .nunique()
    ) > 3E3)
    )

SEP = SEP.merge(
    (SF1
     .query('dimension == "ARQ"')[
        ['ticker', 'datekey', 'shareswa']]
     .rename(columns = {'datekey':'datadate'})
     ),
    how = 'left',
    on = ['ticker', 'datadate'], 
    copy = False)

# Delete KALO, something is wrong in this ticker's data but I don't know what, 
# not worth the time to find out
# Delete ACOL, there is erroneous price data on 2013-04-03, this equity should 
# never be used for anything anyhow
# Both equities were removed from data by Sharadar, keep code in case provider
# repeats the error
SEP = SEP.query("ticker not in ['KALO', 'ACOL']")

SEP['shareswa'] = (
    SEP
    .groupby('ticker')
    .shareswa
    .apply(
        lambda x: 
            (x
            .fillna(method = 'ffill', limit = 150)
            .fillna(method = 'bfill', limit = 50))
            )
    )

SEP['mktCap'] = SEP['close'] * SEP['shareswa']

SEP['shift_mktCap'] = (
    SEP
    .query('close == close')
    .groupby('ticker')
    .mktCap
    .shift()
    )

SEP.loc[SEP.ret.isin([np.inf, -np.inf]), 'ret'] = 0

# Pre-filtering metrics====
#  Drop tickers that would never get used 
# Some companies only report once a year so dimnesion ARQ will not be
#  available. These equities are discarded.
# NB: No market cap is accounted for by fillna
 
# days to being delisted, used in universe filtration 
SEP['days_to_DL'] = (
    SEP
    .groupby('ticker')
    .datadate
    .transform(lambda x: (np.max(x) - x).dt.days)
    )

# Getting book value (SHE), not FF method====
SEP = SEP.merge(
        (SF1
         .query('dimension == "ARQ"')
         [['ticker', 'datekey', 'equityusd']]
         .rename(columns = {'datekey':'datadate', 'equityusd':'SHE'})), 
        on = ['ticker', 'datadate'], 
        how = 'left', 
        copy = False)

SEP['SHE'] = SEP.groupby('ticker').SHE.fillna(method = 'ffill', limit = 150)

# Some financial statements have missing data that was replaced by  0. 
SEP['mtb'] = SEP.mktCap / SEP.query("SHE != 0").SHE

# Rolling accounting data:
SF1_top = (
    SF1
    .query('dimension == "ARQ"')
    .rename(columns = {'datekey':'datadate'})
    )

SF1_top.sort_values(by='datadate', inplace = True)        

SF1_top = SF1_top.groupby(['ticker', 'calendardate']).nth(0).reset_index()

# Fill in missing obs
# Order by calendardate and use datekey as effective date
SF1_dates_full = (
    SF1_top[['ticker']]
    .drop_duplicates()
    .assign(key=1)
    .merge(
        (SF1_top[['calendardate']]
         .drop_duplicates()
         .assign(key=1)), 
        how='outer', 
        on='key', 
        copy=False)
    )

# .map example, to remind what object is using for mapping
# (SF1_top
#  .ticker
#  .map(SF1_top[SF1_top.dimension.notnull()]
#       .groupby('ticker')
#       .calendardate
#       .max())
#  )

SF1_top = SF1_top.merge(
    SF1_dates_full, 
    how='outer', 
    on=['ticker', 'calendardate'], 
    copy=False)

SF1_top = SF1_top.merge(
    (SF1_top
     .query("dimension == dimension")
     .groupby('ticker')
     .calendardate
     .agg(min_cdate = 'min', max_cdate = 'max')
     ), 
    how = 'left', 
    left_on = 'ticker', 
    right_index = True,
    copy = False)

SF1_top = SF1_top.query(
    'calendardate>=min_cdate and calendardate<=max_cdate')

SF1_top.sort_values(['ticker', 'calendardate', 'datadate'], inplace=True)

#  '_12' variables need to be calculated prior to merging with SEP----
SF1_top['ct'] = (
    SF1_top
    .groupby('ticker')
    .calendardate
    .transform(lambda x: x.notnull().sum())
    )

# Technically, this should be normalized by assets or the like
# NB: subsequent similar variables will need a function
# NB: This version doens't fillna in pre 4 periods as 0
SF1_top['ebitda_12'] = (
    SF1_top
    .query('ct > 4')
    .groupby('ticker', as_index=False)
    .ebitdausd
    .rolling(4,closed='right')
    .sum()
    .ebitdausd
    )

# N_datadate is used as an integer for .rolling
SF1_top['N_datadate'] = (
    SF1_top
    .datadate
    .astype(str)
    .str.replace('\D', '', regex = True)
    )

SF1_top.loc[lambda x: x.N_datadate == '', 'N_datadate'] = '19000101'

SF1_top['eff_date'] = (
    SF1_top
    .query('ct > 4')
    .groupby('ticker', as_index=False)
    .N_datadate
    .rolling(4,closed='right')
    .max()
    .N_datadate
    )

SF1_top['eff_date'] = pd.to_datetime(SF1_top.eff_date, format='%Y%m%d')

SF1_top['reportperiod'] = pd.to_datetime(SF1_top.reportperiod)

SF1_top['eff_delta'] = SF1_top.eval('(eff_date-reportperiod).dt.days')

SF1_top = SF1_top.query('eff_delta < 90')

SEP = SEP.merge(
    SF1_top[['eff_date', 'ticker', 'calendardate', 'ebitda_12']], 
    how='left', 
    left_on=['ticker', 'datadate'], 
    right_on=['ticker', 'eff_date'], 
    copy=False)

SEP.sort_values(['ticker', 'datadate'], inplace=True)

SEP['ebitda_12'] = (
    SEP
    .groupby('ticker', group_keys=0)
    .apply(lambda x: x.ebitda_12.shift(1).fillna(method='ffill', limit=110))
    )

SEP['vold15D'] = (
    SEP
    .query('close==close')
    .groupby('ticker', as_index=False)
    .ret
    .rolling(15, closed='right')
    .std(skipna = True)
    .ret
    )

SEP['dollar_vlm'] = SEP.eval('close * volume').replace(0, np.nan)

# NB: not all filters are here due to single factor alpha example
core_vars = {'volume':'volume', 
             'dollar_vlm':'dollar_vlm', 
             # 'sprPerc':'spread', 
             'price':'closeunadj', 
             'mkt_cap':'mktCap', 
             'DL_flag':'days_to_DL', 
             'ebitda':'ebitda_12', 
             'mtb':'mtb', 
             'ret':'ret', 
             'vold':'vold15D'}

coreSEP = {
    var:(SEP
          .query('close==close')[['datadate', 'ticker', name]]
          .pivot_table(values = name, index = 'datadate', columns = 'ticker')
          )
    for var, name in core_vars.items()
    }

del core_vars

df_ret = coreSEP['ret']

# Universe creation========
price_Cutoff = 6        # min price
smallCap = 250E06   # min market cap
mng_port_size = 1E09
grand_univ = (df_ret * np.nan)
prev_tickers=pd.Series([], dtype = str)

SEP['Year'] = SEP.datadate.dt.year
SEP['Month'] = SEP.datadate.dt.month

firstOfMonth = (
    SEP
    .query('close == close')
    .groupby(['Year', 'Month'])
    .datadate
    .nth(0)
    )

starting_date=pd.to_datetime('2009-05-01')

tot_mkt = (
    SEP
    .loc[SEP.datadate.isin(firstOfMonth.values)]
    .groupby('datadate')
    .shift_mktCap
    .sum()
    )

tot_mkt = tot_mkt.loc[tot_mkt.index >= starting_date]
tot_mkt = tot_mkt/tot_mkt[0]
tot_mkt = tot_mkt.rename('mkt_adj')

for UNIV_DATE in firstOfMonth.loc[lambda x: x >= starting_date].values:
    
    UNIV_DATE_30, UNIV_DATE_100, UNIV_DATE_21 = (
        UNIV_DATE 
        - np.array([30, 100, 21], dtype='timedelta64[D]')
        )
    
    UNIV_DATE_1 = (
        SEP
        .query('datadate<@UNIV_DATE & close==close')
        .datadate
        .max()
        )
    
    filter_df = pd.DataFrame({
        # NB: the price column can be used as a starting index for included 
        # equities
        'price':
        (coreSEP['price']
         .loc[UNIV_DATE_30:UNIV_DATE_1,:]
         .mean(axis = 0, skipna = True) > price_Cutoff
         ), 
        
        'mkt_cap':
        (coreSEP['mkt_cap']
         .loc[UNIV_DATE_30:UNIV_DATE_1,:]
         .mean(axis = 0, skipna = True) > smallCap
         ),
        
        'not_delisted':
        coreSEP['DL_flag'].loc[UNIV_DATE_1,:] >= 25, 
        
        'ebitda':
        (coreSEP['ebitda']
         .loc[UNIV_DATE_30:UNIV_DATE_1,:]
         .apply(lambda x: any(x == x))
         ),
        # TODO: mtb needs to be revisted...
        
        'mtb':
        (coreSEP['mtb']
         .loc[UNIV_DATE_30:UNIV_DATE_1,:]
         .apply(
             lambda x: all((x.dropna() > -100) & (x.dropna() < 100)), axis = 0)
         ), 
        # Removes tickers that have very little price movement
        'vold':
        coreSEP['vold'].loc[UNIV_DATE_1,:] > .0065, 
        
        'NoSP500':
        (coreSEP['mkt_cap']
         .loc[UNIV_DATE_30:UNIV_DATE_1,:]
         .mean(axis = 0, skipna = True).rank(ascending = False) > 300
         ),
        # Sufficient observations
        'price_obs':
        (coreSEP['price']
         .loc[UNIV_DATE_100:UNIV_DATE_1,:]
         .apply(lambda x: x.count() >= x.shape[0] * .9)
         ),
        
        'volume_obs':
        ((coreSEP['dollar_vlm']
         .loc[UNIV_DATE_21:UNIV_DATE_1,:] > 0)
         .apply(lambda x: x.sum() >= x.shape[0] * .9)
         )
    })
  
    filter_df = filter_df.fillna(0)
    filter_df['not_delisted'].fillna(value = False, inplace = True)
    
    filter_df['valid'] = filter_df.apply(all, axis = 1)
    cur_ticker=filter_df.query("valid == True").index.values
  
    # Replicate the index in full in 10 days max, assume that other equities  
    # are earning a non-captureable small size premium
    cap_filter = (
        coreSEP['mkt_cap']
        .loc[UNIV_DATE_30:UNIV_DATE_1,cur_ticker]
        .mean(axis = 0, skipna = True)
        .rename('mktCap')
        )
    vlm_filter = (
        coreSEP['dollar_vlm']
        .loc[UNIV_DATE_30:UNIV_DATE_1,cur_ticker]
        .mean(axis = 0, skipna = True)
        .rename('dVlm')
        )
    
    # Average Dollar Volume
    adv = pd.merge(
        cap_filter, 
        vlm_filter, 
        how='inner', 
        left_index=True, 
        right_index=True, 
        copy = False)
    
    adv['w'] = adv.eval('mktCap / mktCap.sum()')
    
    MKT_ADJ = tot_mkt[UNIV_DATE]
    
    #Assumer there is at least 30% more volume in dark pools
    adv['daysToFill'] = adv.eval(
        '@mng_port_size * @MKT_ADJ * w /(dVlm * .01 * 1.3)')
    
    adv['prev'] = adv.index.isin(prev_tickers)
    
    # Valid Set
    vs = (
        adv
        .query('(prev == True & daysToFill <= 5.0 * 1.3) | daysToFill <= 5.0')
        .index
        .values
        )    
    
    # Get the correct slice of dates and account for the end date of the 
    #   very last date in the backtest
    if UNIV_DATE != firstOfMonth.values[-1]:
        END_DATE = (
            firstOfMonth
            .iloc[np.flatnonzero(firstOfMonth.values == UNIV_DATE) + 1]
            .values[0] 
            - np.timedelta64(1, 'D')
            )
    else: 
        END_DATE  = UNIV_DATE + np.timedelta64(30, 'D')

    grand_univ.loc[UNIV_DATE:END_DATE] *= np.nan
    grand_univ.loc[UNIV_DATE:END_DATE, vs] = True
    
    print(str(UNIV_DATE)[:10] + 
          ' ' + 
          str(filter_df.all(axis = 'columns').sum()) + 
          ' \\' + str(vs.shape[0]) + 
          ' Tickers')
     

melt_grand_univ = grand_univ.stack()

melt_grand_univ = melt_grand_univ.to_frame().rename(columns = {0:'is_active'})

SEP = SEP.merge(
    melt_grand_univ, 
    how = 'left', 
    left_on = ['datadate', 'ticker'], 
    right_index = True, 
    copy = False)
# Days to DL < 10 is harder to enforce in practive but necessary
SEP.loc[SEP.days_to_DL < 10, 'is_active'] = False

SEP['ebitda_12_mkt'] = SEP.eval('ebitda_12 / shift_mktCap')

scaler = pre.RobustScaler()

SEP['ebitda_12_mkt_u'] = (
    SEP
    .query('is_active == True')
    .groupby(['datadate', 'sector'])
    .ebitda_12_mkt
    .transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    )

# Optimizer loop
# To get it super right takes a very long time
# Initialize variables:
# start book, end book, delta book
sBook, eBook, dBook = df_ret * 0, df_ret * 0, df_ret.fillna(0) * 0 
volBook, alphaBook = df_ret.AAPL * np.nan, df_ret.AAPL * np.nan
first_date = np.datetime64('2009-05-01')

rebal_dates = df_ret.query("datadate >= @first_date").index.values[::15]

cdate = rebal_dates[0] # Current Date -> cdate
print(cdate)

# Continuous plotting
fig, (ax, ax2) = plt.subplots(
    2, 
    figsize=(12,8), 
    gridspec_kw={'height_ratios': [4, 1]})
ax.set_xlim(rebal_dates[0], rebal_dates[-1])
ax2.set_xlim(rebal_dates[0], rebal_dates[-1])

CUM_RET = 1

LAM = 14500

adj_ct = pd.Series([], dtype = 'float64')

SECTOR_LIM=6E06
   
PORT_TGT = 500E06
TURNOVER_TGT = PORT_TGT / 10

w_TURNOVER = 0

adv_all = (coreSEP['dollar_vlm']
           .rolling(21, closed='right', min_periods = 3)
           .mean()
           .shift())
# Remove the last volume in each name to avoid look ahead bias
stack_adv = adv_all.stack().reset_index().rename(columns = {0:'dVLM'})

stack_adv.loc[
    stack_adv.groupby('ticker').datadate.idxmax().values, 'dVLM'] = np.nan

adv_all = stack_adv.pivot_table(
    index = 'datadate', columns = 'ticker', values = 'dVLM')

sec_dum_all = pd.get_dummies(tickers.query('table == "SEP"').sector)

# TODO: stop if LS diff is greater than max turnover
while cdate <= rebal_dates[-1]:
    # Notes---
    # Assigning long and short ticker constraints allow for controlling 
    # size
    # nas in constraints throw an immediate error
    # Improvement needed to add minimum trade size per name
    # NB: see optimization notes and explanation off github 
    # Need to make sure constraints dont contain contradictory values
    # get full book in 10 days ---> 500/10 = 50 million
    adj_ct = adj_ct.iloc[0:0]
    
    date_1 = cdate - np.timedelta64(1, 'D')
    date_180 = cdate - np.timedelta64(180, 'D')
    
    index_tickers = (
        SEP
        .query('is_active==1 & datadate==@cdate & volume!=0')
        .ticker
        .sort_values()
        .values)
    
    book_names = eBook.loc[cdate].loc[lambda x: (x!=0) & (x==x)].index.values
    
    suspended_tickers = (
        SEP
        .query('datadate==@cdate & volume==0 & ticker in @book_names')
        .ticker
        )
    
    book_names = book_names[~np.isin(book_names, suspended_tickers)]
    
    active_tickers = np.union1d(index_tickers, book_names)
     
    add_tickers = active_tickers[~np.isin(active_tickers, index_tickers)]
    # add_tickers = add_tickers[0:0]
    
    # Notice that SEP is sorted by ticker first
    # sector dummy df
    sec_dum = sec_dum_all.loc[active_tickers]
    
    w = eBook.loc[cdate, active_tickers].rename('wght')
    
    assert ~w.isnull().any()
    
    a = (
        SEP
        .query("datadate == @cdate & ticker in @active_tickers")
        .loc[:, ['ticker', 'ebitda_12_mkt_u']]
        .set_index('ticker')
        .rename(columns = {'ebitda_12_mkt_u':'alpha'})
        .fillna(0)
        )
    
    no_data_adds = active_tickers[~np.isin(active_tickers, a.index)]
    # add tickers that don't have data and are liquidated
    a = (a
         .append(pd.DataFrame(
             data = {'alpha' : np.repeat(0, no_data_adds.size)}, 
             index = no_data_adds))
         .sort_index()
         )
    
    assert(sBook.loc[cdate, no_data_adds].abs().sum() < 300E3)
    assert(no_data_adds.size <= 10)
    
    df_limits = pd.merge(
        w, 
        adv_all.loc[cdate, active_tickers].rename('adv'), 
        left_index = True, 
        right_index = True, 
        how = 'left', 
        copy = False)
    
    df_limits = df_limits.merge(
        (SEP
         .query("datadate==@cdate and ticker in @add_tickers")
         [['ticker', 'days_to_DL']]
         .set_index('ticker')), 
        left_index = True, 
        right_index = True, 
        how = 'left', 
        copy = False)
    
    df_limits['adv13'] = df_limits.adv * .013 # for easy ref
    
    df_limits['pos_limit'] = np.minimum(
        df_limits.eval("10 * adv * .013"), .025 * PORT_TGT)
    
    # double the limit when the name is scheduled for delisting
    df_limits.loc[df_limits.days_to_DL <= 10, 'adv13'] *= 2
    
    df_limits.loc[no_data_adds, 'adv13'] = df_limits.wght.abs() * 1.1
    
    df_limits['liq'] = (
        (df_limits
        .loc[add_tickers]
        [['wght', 'adv13']]
        .abs()
        .min(axis = 1)) 
        * -np.sign(df_limits.wght)
        )
    
    full_optim = (
        (cdate in rebal_dates) or 
        (np.abs(PORT_TGT - eBook.loc[cdate].abs().sum()) > (PORT_TGT * .1)) #or 
        # Higher turnover condition
        # wSol.abs().sum() > (PORT_TGT * .1) or 
        # np.max((sec_dum.T.dot(w).abs().values)) > (SECTOR_LIM * 1.2)
        )
            
    if full_optim:
        df_limits['semi_buy_limit'] = np.minimum(
            df_limits.eval("pos_limit - wght"), df_limits.adv13)
        
        df_limits['semi_sell_limit'] = np.maximum(
            df_limits.eval("-pos_limit - wght"), -df_limits.adv13)
        
        df_limits['buy_limit'] = np.maximum(df_limits.semi_buy_limit, 0)
        df_limits['sell_limit'] = np.minimum(df_limits.semi_sell_limit, 0)
        
        df_limits.loc[df_limits.buy_limit == 0, 'buy_limit'] = df_limits.adv13
        df_limits.loc[df_limits.sell_limit == 0, 'sell_limit']=-df_limits.adv13
        
        df_limits.loc[add_tickers, 'buy_limit'] = df_limits.liq.abs()*1.1
        df_limits.loc[add_tickers, 'sell_limit'] = -df_limits.liq.abs()*1.1
        
        df_limits['buy_min'] = (
            df_limits
            .query("(sell_limit == 0) and ticker not in @add_tickers")
            [['semi_sell_limit', 'adv13']]
            .min(axis = 1)
            )
        
        df_limits['sell_min'] = (
            df_limits
            .query("(buy_limit == 0) and ticker not in @add_tickers")
            [['semi_buy_limit', 'adv13']]
            .transform({'semi_buy_limit': lambda x: x, 'adv13':lambda x: -x})
            .max(axis = 1)
            )
        
        # To avoid rounding errors in optimization
        df_limits.loc[add_tickers, ['buy_limit', 'sell_limit']] *= 1.1
        
        liq_matrix = np.zeros((add_tickers.size, w.size))
        # Accepting tips on how to replace the for loop with something prettier
        for i in np.arange(add_tickers.size):
            liq_matrix[i, np.where(w.index.values == add_tickers[i])] = 1
        
    df_limits['sector'] = tickers.query('table == "SEP"').sector
    
    df_limits['sector_lim_w'] = (
        df_limits
        .sector
        .map(
            np.minimum(
                1, 
                (SECTOR_LIM * 1.2) 
                / (df_limits
                   .query('ticker not in @add_tickers')
                   .groupby('sector')
                   .wght.sum().abs()
                   )
                )
            )
        )
    
    df_limits['liq_sector_w'] = (
        df_limits
        .query('sector_lim_w != 1')
        .eval('-1 * wght * (1 - sector_lim_w)')
        )
    
    # minimum set of wSol, added trimming excess sector positions
    wSol = df_limits.liq.fillna(df_limits.liq_sector_w).fillna(0)
    
    diag_mat = np.eye(len(active_tickers))
    
    cov_mat = df_ret.loc[date_180:date_1, active_tickers].cov()
    # shrink a little bit to promote invertibility, new Covariance Matrix
    nCM = (.1 * np.diag(cov_mat).mean() * np.eye(len(active_tickers))) \
          + (.90 * cov_mat)
    
    mu=.015/2/2
    
    if wSol.abs().sum() > (PORT_TGT * .1): 
        print('~~~~~~ OUT OF UNIVERSE NAMES ARE MATERIAL ~~~~~~~')
        
    while ((np.abs(w_TURNOVER - TURNOVER_TGT) > (.1 * TURNOVER_TGT)) or 
          adj_ct.size == 0):
        if not full_optim: break
        # Lost a little efficiency/ PEP8 for much better readability
        # a few variables that don't change get reinitialized for debugging
        
        tau = .01E-02 # this is where t-cost comes in if datea is available
        
        P = (2 * mu * 
            np.hstack((np.vstack((nCM, nCM)), 
                       np.vstack((nCM, nCM)))))
        
        q = np.hstack((2 * mu * nCM.dot(w) - a.alpha + LAM * tau, 
                       2 * mu * nCM.dot(w) - a.alpha - LAM * tau))
        
        G = np.vstack(
            (
                np.hstack((-1 * diag_mat, 0 * diag_mat)),
                np.hstack(( 0 * diag_mat, 1 * diag_mat)), 
                
                np.hstack((1 * diag_mat,  0 * diag_mat)),
                np.hstack((0 * diag_mat, -1 * diag_mat)), 
                
                np.hstack((sec_dum.T, sec_dum.T)), 
                np.hstack((-1*sec_dum.T, -1*sec_dum.T))           
            ))
        
        h = np.hstack((
            # Bind y and z to be semi-poistive and semi-negative
            -df_limits['buy_min'].fillna(0),
            df_limits['sell_min'].fillna(0),
            # Daily limit and position size limits
            df_limits['buy_limit'],
            -df_limits['sell_limit'], 
            # Sector neutrality limits
            SECTOR_LIM * np.ones(sec_dum.shape[1]) - sec_dum.T.dot(w).values, 
            SECTOR_LIM * np.ones(sec_dum.shape[1]) + sec_dum.T.dot(w).values, 
            ))
        
        A = np.vstack(
            (np.ones([1, len(active_tickers)*2]), 
             # np.hstack((liq_matrix, liq_matrix))
             np.hstack((1 * liq_matrix, 0 * liq_matrix)),
             np.hstack((0 * liq_matrix, 1 * liq_matrix)))
            )
        
        b = np.concatenate(
                (0 - w.sum(), # Enforce strict long - short portfolio
                 np.maximum(df_limits.liq.dropna().values, 0),
                 np.minimum(df_limits.liq.dropna().values, 0)),
                axis = None)
        
        H_SCALER = np.round(pd.Series(h).mean() / 500, -2)
        # # debug all inputs
        # for i in ['P', 'q', 'G', 'h', 'A', 'b']:
        #     print(pd.DataFrame(globals()[i]).isnull().sum().sum(), '\n')
        
        P_ = cvxopt.matrix(P, tc='d')
        q_ = cvxopt.matrix(q, tc='d')
        G_ = cvxopt.matrix(G / H_SCALER, tc='d')
        h_ = cvxopt.matrix(h / H_SCALER, tc='d')
        A_ = cvxopt.matrix(A, tc='d')
        b_ = cvxopt.matrix(b, tc='d')
        
        sol = cvxopt.solvers.qp(P_,q_,G_,h_,A_,b_, maxiters = 30)
        
        assert sol['status'] == 'optimal'
        # assert np.abs(sol['gap']) < 3E4
    
        wSol = np.array(sol['x']).reshape((2, -1)).T.sum(axis = 1).round(2)
        
        w_TURNOVER = np.abs(wSol).sum().round(2)
        
        adj_ct = adj_ct.append(pd.Series({w_TURNOVER:LAM})).rename('lam0')
        
        print(
            'tried LAM = ', LAM, 
            '...Turnover: ', np.round(w_TURNOVER/1E6, 1),'\bE6 \n')
        
        if (np.abs(w_TURNOVER - TURNOVER_TGT) > (.1 * TURNOVER_TGT)):
            
            if adj_ct.size > 5: 
                raise ValueError(
                    "Sufficient lambda value not found too many times")

            # Use linear interpolation! No need for second order expansion
            if adj_ct.size == 1:
                LAM *= w_TURNOVER / TURNOVER_TGT 
            else: 
                LAM = (
                    adj_ct.iloc[-1] 
                    + (
                        (adj_ct.iloc[-1] - adj_ct.iloc[-2]) 
                        / (adj_ct.index[-1] - adj_ct.index[-2])
                        ) 
                    * (TURNOVER_TGT - adj_ct.index[-1])
                    )
            if LAM<1 and adj_ct.min()>1: LAM = 1
            LAM = np.round(LAM)
            
            if LAM < 0: raise ValueError('lambda is negative!')
            
        
    if wSol[wSol != 0].size > add_tickers.size:
        print(f'LAM is: {LAM} || ',
              f'w_TURNOVER: {"{:0=.2f}".format(w_TURNOVER/1E6)}E06 || ',
              f'cdate: {str(cdate)[:10]} \n')
    
    # Rebalance happens at the end of the day
    if wSol[wSol != 0].size != 0:
        
        df_wSol = pd.merge(
            eBook.loc[cdate, active_tickers].rename('book'), 
            pd.DataFrame({'adds' : wSol}, index = active_tickers),
            left_index = True, 
            right_index = True, 
            how = 'right', 
            copy = False)
        
        df_wSol.rename_axis(index='ticker', inplace = True)
        # The !=0 is for debugging, it doesn't do anything
        df_wSol.loc[
            (~df_wSol.index.isin(add_tickers)) & 
            (df_wSol.adds.abs() < 10E3) & 
            (df_wSol.adds != 0), 
            'adds'] = 0 
        
        df_wSol['check_book'] = df_wSol.eval("adds + book")
        
        # For extra certainty
        df_wSol.loc[
            (df_wSol.check_book.abs() < 1E3) & (df_wSol.check_book != 0), 
            'check_book'] = 0
        
        df_wSol['sector'] = tickers.query('table == "SEP"').sector
        
        df_wSol['sector_w'] = (
            (df_wSol.
             groupby('sector').
             check_book.
             transform(lambda x: x.abs().sum()))
            / df_wSol.check_book.abs().sum()
            )
        
        book_tgt = (
            .5 
            * (np.min(
                (PORT_TGT - df_limits.query('liq != liq').wght.abs().sum(), 
                TURNOVER_TGT)) # easier way to rewrite the below?
               + (w.abs().sum() 
                  - (df_limits
                     .query('liq == liq')
                     .eval('wght - liq')
                     .abs()
                     .sum()))
               )
        )
        
        df_wSol['final_book'] = (
            df_wSol
            .query("check_book != 0 & ticker not in @add_tickers")
            .groupby(np.sign(df_wSol.check_book), group_keys = False)
            # [['check_book', 'sector']]
            .apply(lambda x: x.check_book * (book_tgt / x.check_book.abs().sum()))
            .round(1)
            )
        # debug leftover
        # df_wSol[df_wSol.columns.difference(['sector'])].abs().sum()
        
        df_wSol.final_book.fillna(df_wSol.check_book, inplace = True)
        df_wSol['delta'] = df_wSol.eval("final_book - book")
        
        assert ~df_wSol.isnull().any().any()
        
        dBook.loc[cdate] = df_wSol['delta']
    
    idate = np.argwhere(df_ret.index.values == cdate)[0] #integer date
    
    ndate=df_ret.index.values[idate[0]+1]
    # The slice must contain all names due to errors if dates need to be rerun
    sBook.loc[ndate] = (eBook.loc[cdate] + dBook.loc[cdate]).fillna(0)
    
    # E[volatility]
    volBook[cdate] = (
        sBook.loc[ndate, active_tickers]
        .dot(nCM)
        .dot(sBook.loc[ndate, active_tickers].T)
        )
    # E[alpha]
    alphaBook[cdate] = sBook.loc[ndate, active_tickers].dot(a).values
    
    eBook.loc[ndate] = sBook.loc[ndate] * (1 + df_ret.loc[ndate].fillna(0))
    
    print(
        f'port size: {"{:0=.2f}".format(sBook.loc[ndate].abs().sum()/1E6)}E06',
        f' trading: {"{:0=.2f}".format(dBook.loc[cdate].abs().sum()/1E6)}E06',
        f' cdate: {str(cdate)[:10]} \n')
        
    next_ret = (
        (eBook.loc[ndate].sum() - sBook.loc[ndate].sum()) 
        / (.3 * sBook.loc[ndate].abs().sum())
        )
    
    assert np.abs(next_ret) < .1
    
    # Real time plotting
    color = '#2CA453'
    if next_ret < 0: color= '#F04730'
    
    ax.plot([cdate, ndate], [CUM_RET, CUM_RET * (1 + next_ret)], color = color)
    ax2.bar(ndate, sBook.loc[ndate].abs().sum(), color='lightblue')
    
    CUM_RET *= 1 + next_ret
    cdate = ndate
    # for plot update
    plt.pause(0.05)
    

# https://towardsdatascience.com/
# downloading-historical-stock-prices-in-python-93f85f059c1f1
Symbols = ['SPY', 'IWM', 'AAPL']
START_DATE = SEP.datadate.min()
END_DATE = SEP.datadate.max()
BM = pd.DataFrame()
# iterate over each symbol
for i in Symbols:  
    # print the symbol which is being downloaded
    print(str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)  
    try:
        # download the stock price 
        stock = []
        stock = yf.download(i,start=START_DATE, end=END_DATE, progress=False)
        # append the individual stock prices 
        if len(stock) == 0:
            None
        else:
            stock['Name']=i
            BM = BM.append(stock,sort=False)
    except Exception:
        None    


group_dates = pd.DataFrame({'datadate' : df_ret.index.values})

group_dates['rebal_date'] = group_dates.query("datadate in @rebal_dates").datadate

group_dates.rebal_date = group_dates.rebal_date.shift()

group_dates.rebal_date = group_dates.query("rebal_date == rebal_date").datadate

group_dates.rebal_date = group_dates.rebal_date.fillna(method = 'ffill')

SEP['rebal_date'] = SEP.datadate.map(group_dates.set_index('datadate').squeeze())

# Speeding up queries
SEP['any_active'] = (
    SEP
    .ticker
    .isin(SEP[['ticker', 'is_active']].query('is_active == True').ticker.unique())
    )

agg_ret = (
    SEP
    [['datadate', 'rebal_date', 'ticker', 'ret', 'any_active', 
      'is_active', 'shift_mktCap', 'ebitda_12_mkt_u', 'sector']]
    .query("rebal_date==rebal_date & any_active==True")
    )

agg_ret['rebal_active'] = (
    agg_ret
    .groupby(['rebal_date', 'ticker'])
    .is_active
    .fillna(method = 'ffill')
    )

agg_ret = agg_ret.query('rebal_active == rebal_active')

agg_ret['alpha_quintile'] = (
    agg_ret
    .query("is_active==True & rebal_date==datadate")
    .groupby(['datadate', 'sector'])
    .ebitda_12_mkt_u
    .transform(lambda x: pd.qcut(x, 5, labels = False))
    )

# This is beginning of the day rebal vs strat end of day
agg_ret['BM_w'] = (
    agg_ret
    .query('is_active==True & rebal_date == datadate')
    .groupby('rebal_date', group_keys = False)
    .apply(lambda x: x.shift_mktCap / x.shift_mktCap.sum())
    )

agg_ret['quintile_w'] = (
    agg_ret
    .query('is_active==True & rebal_date == datadate')
    .groupby(['rebal_date', 'alpha_quintile'], group_keys = False)
    .apply(lambda x: x.shift_mktCap / x.shift_mktCap.sum())
    )

agg_ret[['BM_w', 'alpha_quintile', 'quintile_w']] = (
    agg_ret
    .groupby(['ticker', 'rebal_date'])
    [['BM_w', 'alpha_quintile', 'quintile_w']]
    .fillna(method = 'ffill')
    )
    
piv_agg_ret = agg_ret.pivot(
    index = ['datadate', 'ticker', 'BM_w', 'ret'], 
    columns = ['alpha_quintile'], 
    values = 'quintile_w')

piv_agg_ret.drop(columns = np.nan, inplace = True)

piv_agg_ret.rename(columns = lambda x: 'quintile_' + str(x)[0], inplace = True)

piv_agg_ret.reset_index(['BM_w', 'ret'], inplace = True)

# Long Short Plotting
BM = BM.query("Name == 'IWM'")
BM.rename(columns = {'Adj Close' : 'adj_close'}, inplace = True, copy = False)
BM['BM_ret'] = BM.adj_close / BM.adj_close.shift() - 1

backtest = reduce(
    lambda left_df, right_df: 
        pd.merge(left_df, right_df, how = 'left', on = 'datadate'), 
    [eBook.sum(axis = 1).rename('end_book'), 
     sBook.sum(axis = 1).rename('start_book'), 
     sBook.abs().sum(axis = 1).rename('book_size'),
     eBook.abs().sum(axis = 1).rename('abs_book_size'), 
     # Long side only
     np.maximum(0, eBook).sum(axis = 1).rename('end_book_long'), 
     np.maximum(0, sBook).sum(axis = 1).rename('start_book_long'), 
     np.maximum(0, sBook).abs().sum(axis = 1).rename('book_size_long')])
    
backtest = backtest.query('datadate >= @rebal_dates[0]')

backtest['ret'] = backtest.eval("(end_book - start_book) / (.3 * book_size)")
backtest['ret_long'] = backtest.eval(
    "(end_book_long - start_book_long) / (book_size_long)")

backtest = backtest.query("(book_size != 0) or (datadate < @rebal_dates[-1])")

backtest.loc[rebal_dates[0]].fillna(0, inplace = True)

LS_plot_data = (
    backtest
    .ret
    .rename('LS_Backtest')
    .to_frame()
    .merge(
        (piv_agg_ret
         [['BM_w'] + ['quintile_' + str(x) for x in range(0,5)]]
         .apply(lambda x: x * piv_agg_ret.ret)
         .groupby('datadate')
         .sum()
         .eval('quintile_4 - quintile_0')
         .rename('LS_Quintiles')),
        how = 'left', 
        left_index = True, 
        right_index = True, 
        copy = False)
    )

LS_plot_data.iloc[0] = LS_plot_data.iloc[0].fillna(0)

# https://towardsdatascience.com/
# basics-of-ohlc-charts-with-pythons-matplotlib-56d0e745a5be
fig, (ax, ax2) = plt.subplots(
    2, 
    figsize=(12,8), 
    gridspec_kw={'height_ratios': [4, 1]})

ax.plot((1+ LS_plot_data).cumprod(), label = LS_plot_data.columns)
ax2.bar(backtest.index, backtest.abs_book_size / 1E6, color='lightblue')
ax2.plot(backtest.abs_book_size / 1E6, color='green')
ax.legend()
ax.margins(x = .02)
ax2.margins(x = .02)
ax.set_ylabel('Portfolio Growth')
ax2.set_ylabel('Port Size\n(in $ millions)')# grid
ax2.set_yticks(np.arange(0,7) * 100)
ax.grid(color='black', linestyle='dashed', which='both', alpha=0.2)
ax2.set_axisbelow(False)
ax2.grid(color='black', linestyle='dashed', which='both', alpha=0.2)# remove spines
ax.set_title('Long-Short Portfolios', color = 'darkgreen')
ax2.set_xlabel('Date')

# Benchmarks Comparison Plotting
BM_plot_data =  reduce(
    lambda left_df, right_df: 
        pd.merge(
            left_df, 
            right_df, 
            how = 'left', 
            left_index = True, 
            right_index = True), 
    [backtest.ret_long.rename('Backtest_long_leg'),
     piv_agg_ret.eval('BM_w*ret').groupby('datadate').sum().rename('Universe'),
     BM.BM_ret.rename('IWM')])
 
BM_plot_data.iloc[0] = 0

((1 + BM_plot_data)
.cumprod()
.plot())

plt.grid('on', ls = '--')
plt.ylabel('Portfolio Value')
plt.xlabel('Date')
plt.title('Comparison to Long Benchmarks')
