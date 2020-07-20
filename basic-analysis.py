from astropy import table 
import numpy as np

def average_growth(x, freq=1):
    return 100 * (np.prod( (1 + x/100) ** (freq / len(x))) - 1)

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
    bin_number = (tab['YEAR'] < bins[:,None]).sum(axis=0) - 1
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
    filenames = ['population-growth.csv', 
                 'elderly.csv',
                 'tax-revenue.csv', 
                 'public-debt.csv',
                 'gdp-growth.csv'
                ]
    bins = [1998, 2008, 2018]
    tab = join(filenames) 
    binned = bin(tab, bins=[1998, 2008, 2018])
