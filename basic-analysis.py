#! /usr/bin/env python3

from astropy import table 
import numpy as np
from scipy.stats import linregress
from matplotlib import pylab as plt
from scipy.optimize import curve_fit

def average_growth(x, freq=1):
    return 100 * (np.prod( (1 + x/100) ** (freq / len(x))) - 1)

def linear_model(X, *p):
    return p[0] + np.dot(X, p[1:])

def kappa_sigma_regress(x, y, kappa=3.0):
    keep = np.ones((len(y),), dtype=bool)
    for i in range(3):
        xp, yp = x[keep], y[keep]
        reg = linregress(xp, yp)
        r = y - (reg.slope * x + reg.intercept)
        keep &= abs(r) <= kappa * r[keep].std()
    return reg, keep
    
def kappa_sigma_multiregress(X, y, kappa=3.0):
    keep = np.ones((len(y),), dtype=bool)
    p0 = np.zeros((X.shape[1]+1,))
    for i in range(3):
        p, pcov = curve_fit(linear_model, X[keep], y[keep], p0=p0)
        res = y - linear_model(X, *p)
        keep &= abs(res) <= kappa * res[keep].std()
    return p, pcov, keep

def parse_table(tab, short_names=False):
    columns = ['INDICATOR', 'SUBJECT', 'MEASURE']
    # OECD has table mixing indicators or measures, ensure it's not the case
    for col in [*columns, 'FREQUENCY']:
        assert len(np.unique(tab[col])) == 1, 'table not consistent'
    # quaterly data are skimmed to last quarter
    if tab['FREQUENCY'][0] == 'Q':
        keep = np.array([t[-2:] == 'Q4' for t in tab['TIME']])
        # if quaterly growth is given, compute yearly one
        if tab['MEASURE'][0] in ['QGROWTH', 'PC_CHGPQ']:
            groups = tab.group_by(['LOCATION', 'TIME'])
            for g in groups:
                g[-1]['Value'] = average_growth(g['Value'], freq=4) 
        keep = np.array([t[-2:] == 'Q4' for t in tab['TIME']])
        tab = tab[keep]
        year = np.array([int(t[0:4]) for t in tab['TIME']])
        tab.remove_column('TIME')
        tab.add_column(year, name='TIME')
        if tab['INDICATOR'][0][0] == 'Q':
            tab['INDICATOR'] = tab['INDICATOR'][0][1:]
    if short_names:
        tab['Value'].name = tab[columns[0]][0]
    else:
        tab['Value'].name = '_'.join([tab[col][0] for col in columns])
    tab.remove_columns([*columns, 'FREQUENCY', 'Flag Codes'])
    return tab

def join(tabs, keys=['LOCATION', 'TIME']):
    if isinstance(tabs[0], str):
        tabs = [table.Table.read(f, encoding='utf-8-sig') for f in filenames]
    tabs = [parse_table(tab) for tab in tabs]
    tab = tabs[0]
    for t in tabs[1:]:
        tab = table.join(tab, t, keys=keys)
    tab['TIME'].name = 'YEAR'
    return tab 

def average_column(col, binsize):
    name = col.name
    if 'CHGPY' in name or 'GROWTH' in name:
        def avg(x): return average_growth(x, freq=1)
    elif col.name in ['LOCATION', 'BINSIZE', 'YEARS']:
        def avg(x): return x[0]
    else:
        def avg(x): return np.mean(x)
    return [avg(g) for g, g2 in zip(col.groups, binsize.groups) 
                                                        if g2[0] == len(g)]
 
def bin(tab, bins):
    # define the bins and add columns
    bins = np.array(bins)
    nbins = len(bins) - 1
    bin_number = (tab['YEAR'] >= bins[:,None]).sum(axis=0) - 1 
    keep = (bin_number >= 0) * (bin_number < nbins)
    tab = tab[keep]
    bin_number = bin_number[keep]
    bin = [f"{bins[b]}-{bins[b+1]-1}" for b in bin_number] 
    binsize = bins[bin_number + 1] - bins[bin_number] 
    tab.remove_column('YEAR')
    tab.add_column(bin, name='YEARS', index=1)
    tab.add_column(binsize, name='BINSIZE')
    # do the rebinning
    tab = tab.group_by(['LOCATION', 'YEARS'])
    cols = [average_column(c, tab['BINSIZE']) 
                for c in tab.columns.values() if c.name != 'BINSIZE']
    names = [c for c in tab.colnames if c != 'BINSIZE']
    tab = table.Table(cols, names=names)
    return tab
    

if __name__ == "__main__":
    filenames = ['oecd/population-growth.csv', 
                 'oecd/elderly.csv',
                 'oecd/tax-revenue.csv', 
                 'oecd/public-debt.csv',
                 'oecd/gdp.csv',
                 'oecd/gdp-growth.csv',
                ]
    bins = [1996, 2003, 2012, 2018]
    tab0 = join(filenames) 
    tab = bin(tab0, bins=bins)
    G = ((
                    (1+tab['GDP_TOT_PC_CHGPY']/100) 
                  / (1+tab['POP_TOT_AGRWTH']/100)
                 ) - 1) * 100
    variables = {
        'public debt [%GDP]': tab['GGDEBT_TOT_PC_GDP'],
#         'elderly population [%total]': tab['ELDLYPOP_TOT_PC_POP'],
        'tax revenue [%GPD]': tab['TAXREV_TOT_PC_GDP'],
        'log GDP [log USD]': np.log10(tab['GDP_TOT_USD_CAP'])
    }    
    fig = plt.figure(1, figsize=(8, 6))
    fig.clf()
    for i, var in enumerate(variables):
        ax = fig.add_subplot(2, 2, i + 1)
        x = variables[var]
        for years in np.unique(tab['YEARS']):
            keep = tab['YEARS'] == years
            xp, yp = x[keep], G[keep]
            (a, b, r, p, e), kept = kappa_sigma_regress(xp, yp)
            nclipped = len(kept) - sum(kept)
            print(f"{var} {years} {a:6.3f}±{e:5.3f} [p-value {p:5.3f}, clipped {nclipped}]")
            scatter = ax.plot(xp[kept], yp[kept], 'o', label=years)[0]
            color = scatter.get_markerfacecolor()
            ax.plot(xp[~kept], yp[~kept], 'o', color=color, alpha=0.2)
            xr = np.array([min(xp), max(xp)])
            yr = a * xr + b
            ax.plot(xr, yr, '-', color=color)
        ax.legend()
        ax.set_xlabel(var)
        ax.set_ylabel('GDP/capita growth [%/year]')
    print('---')
    for year in np.unique(tab['YEARS']):
        keep = tab['YEARS'] == year
        X = np.array([v for v in variables.values()]).T
        Xp, yp = X[keep], G[keep]
        p, pcov, kept = kappa_sigma_multiregress(Xp, yp)
        A = p[1:] # coeff
        dA = np.sqrt(pcov.diagonal()[1:])
        nclipped = len(kept) - sum(kept)
        for var, a, da in zip(variables, A, dA):
            print(f"{year} {var} {a:6.3f}±{da:5.3f} [clipped: {nclipped}]")
        print('---')

