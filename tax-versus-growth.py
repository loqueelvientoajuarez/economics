# countries with tax burdens between 10 and 50% are considered 
tmin = 10
tmax = 50

from astropy.table import Table, join
import numpy as np
from scipy.stats import linregress
from matplotlib import pylab as plt

tax = Table.read('oecd/tax-revenue.csv', encoding='utf-8-sig') 
pop = Table.read('oecd/population-growth.csv', encoding='utf-8-sig')
gdp = Table.read('oecd/gdp-growth.csv', encoding='utf-8-sig')

# convert quartely to yearly
gdp_year = [int(t[0:4]) for t in gdp['TIME']]
gdp.add_column(gdp_year, name='YEAR')
groups = gdp.group_by(['LOCATION', 'YEAR']).groups
cols = [[g['LOCATION'][0] for g in groups],
        [g['YEAR'][0] for g in groups],
        [g['Value'][-1] for g in groups]]
names = ['LOCATION', 'TIME', 'gdp_Value']
gdp = Table(cols, names=names)

# join tables by year
tab = join(tax, pop, keys=['LOCATION', "TIME"], 
                table_names=["tax", "pop"], 
                uniq_col_name='{table_name}_{col_name}'
          )
tab = join(tab, gdp, keys=['LOCATION', "TIME"])

# average by decade
tab.add_column(tab['TIME'] // 10 * 10, name='DECADE')
tab = tab.group_by(['LOCATION', 'DECADE']) 
groups = tab.groups
cols = [[g['LOCATION'][0] for g in groups],
        [g['DECADE'][0] for g in groups],
        [100*(np.prod(1 + g['gdp_Value']/100)**(1/len(g))-1) for g in groups],
        [100*(np.prod(1 + g['pop_Value']/100)**(1/len(g))-1) for g in groups],
        [g['tax_Value'].mean() for g in groups]]
names = ['LOCATION', 'DECADE', 'GDP_GROWTH', 'POP_GROWTH', 'TAX_BURDEN']

tab = Table(cols, names=names)
 
# plotting

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111)
A = []
E = []
REJECTED = []

# linear regression

# data too scarce in the 60s
DECADES = np.unique(tab['DECADE'][tab['DECADE'] > 1960]) 
for decade in DECADES:
    keep = tab['DECADE'] == decade
    tax = tab['TAX_BURDEN'][keep]
    gdp_growth = tab['GDP_GROWTH'][keep] 
    pop_growth = tab['POP_GROWTH'][keep]
    growth = ((1 + gdp_growth / 100)  / (1 + pop_growth / 100) - 1) * 100
    # 3σ-clipping
    not_clipped = True 
    for i in range(3):
        keep = (tax <= tmax) * (tax >= tmin) * not_clipped
        a, b, c, d, e = linregress(tax[keep], growth[keep])
        dy = growth - (a * tax + b)
        not_clipped = abs(dy) <= dy[keep].std() * 3
    nclip = sum(~not_clipped)
    A.append(a)
    E.append(e)
    scatter = ax.plot(tax[keep], growth[keep], 'o', label=f"{decade}s")
    color = scatter[0].get_markerfacecolor()
    scatter = ax.plot(tax[~keep], growth[~keep], 'o', color=color,
        alpha=.33)
    x = np.linspace(max(tmin, tax[keep].min()), min(tmax, tax[keep].max()))
    y = a * x  + b
    REJECTED.append(nclip)
    ax.plot(x, y, '-', color=color)

E = np.array(E)
A = np.array(A)
Amean = np.average(A, weights=1/E**2)
Aerr = np.sqrt(1/sum(1/E**2))

# summary of findings

print(f"""Growth rate of GDP/capita (%/year) v. tax burden (% GDP) averaged by decade. 

Data origin OECD website:
    * GDP growth: https://data.oecd.org/gdp/quarterly-gdp.htm 
    * population growth: https://data.oecd.org/pop/population.htm
    * tax revenue: https://data.oecd.org/tax/tax-revenue.htm
 
Regression with  3σ-clipping.

The impact of 10% GDP of tax is {10*Amean:5.2f}±{10*Aerr:4.2f}% growth per year, the weighted average of the estimates for each decade:""")

for d, a, e, r  in zip(DECADES, A, E, REJECTED):
    print(f"    {d}s:  {10*a:5.2f}±{10*e:4.2f}% [{r} point(s) rejected]")

# plots
plt.style.use('fivethirtyeight')
ax.legend()
ax.set_xlabel('tax burden [%GDP]')
ax.set_ylabel('GDP/capita increase [%/year]')
ax2 = ax.twiny()
ax2.set_xticks([])
ax2.set_xlabel('OECD growth vs. tax burden by decade and country')
ax.set_xlim(10,50)
fig.tight_layout()
fig.show()
fig.savefig('tax.png')
