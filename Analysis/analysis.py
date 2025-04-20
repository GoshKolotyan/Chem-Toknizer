import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create results directory if it doesn't exist
results_dir = '../Results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Add timestamp to filenames to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_elements_from_formula(formula, pattern, element_sites):
    """Extract elements from a chemical formula and categorize them by site."""
    matches = pattern.findall(formula)
    elements_by_site = {'A-site': [], 'B-site': [], 'X-site': []}
    
    for match in matches:
        # Skip numbers
        if match.replace('.', '').isdigit():
            continue
        
        # Categorize element by site
        if match in element_sites:
            site = element_sites[match]
            elements_by_site[site].append(match)
    
    return elements_by_site

def find_most_common_elements(data, pattern, element_sites):
    """Find the most common elements for each site in the dataset."""
    # Create counters for each site
    a_site_counter = Counter()
    b_site_counter = Counter()
    x_site_counter = Counter()
    
    # Create counters for compositions (including double elements)
    a_site_composition_counter = Counter()
    b_site_composition_counter = Counter()
    x_site_composition_counter = Counter()
    
    # Process each formula in the dataset
    for formula in data:
        elements_by_site = extract_elements_from_formula(formula, pattern, element_sites)
        
        # Update counters for individual elements
        a_site_counter.update(elements_by_site['A-site'])
        b_site_counter.update(elements_by_site['B-site'])
        x_site_counter.update(elements_by_site['X-site'])
        
        # Count compositions (including doubles)
        # For A-site
        if elements_by_site['A-site']:
            if len(elements_by_site['A-site']) == 1:
                a_site_composition_counter[elements_by_site['A-site'][0]] += 1
            else:
                # Sort elements to ensure consistent ordering
                sorted_composition = ''.join(sorted(elements_by_site['A-site']))
                a_site_composition_counter[sorted_composition] += 1
        
        # For B-site
        if elements_by_site['B-site']:
            if len(elements_by_site['B-site']) == 1:
                b_site_composition_counter[elements_by_site['B-site'][0]] += 1
            else:
                sorted_composition = ''.join(sorted(elements_by_site['B-site']))
                b_site_composition_counter[sorted_composition] += 1
                
        # For X-site
        if elements_by_site['X-site']:
            if len(elements_by_site['X-site']) == 1:
                x_site_composition_counter[elements_by_site['X-site'][0]] += 1
            else:
                sorted_composition = ''.join(sorted(elements_by_site['X-site']))
                x_site_composition_counter[sorted_composition] += 1
    
    return (a_site_counter, b_site_counter, x_site_counter, 
            a_site_composition_counter, b_site_composition_counter, x_site_composition_counter)

# Load element classification data
known_A_elements = pd.read_csv('../Data/A_part_AA.csv', delimiter=';')
known_B_elements = pd.read_csv('../Data/B_part_AA.csv', delimiter=';')
known_X_elements = pd.read_csv('../Data/X_part_AA.csv', delimiter=';')

known_A_symbols = list(known_A_elements['Element'])
known_B_symbols = list(known_B_elements['Element'])
known_X_symbols = list(known_X_elements['Element'])

# Create element_sites dictionary
element_sites = {}
for element in known_A_symbols:
    element_sites[element] = "A-site"
for element in known_B_symbols:
    element_sites[element] = "B-site"
for element in known_X_symbols:
    element_sites[element] = "X-site"

multi_letter_symbols = known_A_symbols + known_B_symbols + known_X_symbols
sorted_multi_symbols = sorted(multi_letter_symbols, key=len, reverse=True)
multi_letter_pattern = "|".join(sorted_multi_symbols)

pattern = re.compile(
    rf'(?:{multi_letter_pattern})|[A-Z][a-z]?|\d+(?:\.\d+)?'
)

# Load perovskite data
mpd_data = pd.read_csv('../Data/perovskite_normalized_unique.csv')
our_data = pd.read_csv('../Data/Full_Data.csv')[["Name","BandGap"]]

# Combine names from both datasets
names = mpd_data['Name'].values.tolist() + our_data['Name'].values.tolist()
names = list(set(names))

# Find most common elements and compositions
(a_counter, b_counter, x_counter, 
 a_comp_counter, b_comp_counter, x_comp_counter) = find_most_common_elements(names, pattern, element_sites)

# Create DataFrames for individual elements
element_data = {
    'Site': [],
    'Element': [],
    'Count': [],
    'Percentage': []
}

# Calculate total counts for percentages
a_total = sum(a_counter.values())
b_total = sum(b_counter.values())
x_total = sum(x_counter.values())

# Fill the element data
for site, counter, total in [('A-site', a_counter, a_total), 
                           ('B-site', b_counter, b_total), 
                           ('X-site', x_counter, x_total)]:
    for element, count in counter.most_common():
        element_data['Site'].append(site)
        element_data['Element'].append(element)
        element_data['Count'].append(count)
        element_data['Percentage'].append(round(count/total*100, 2) if total > 0 else 0)

element_df = pd.DataFrame(element_data)

# Create DataFrames for compositions
composition_data = {
    'Site': [],
    'Composition': [],
    'Count': [],
    'Type': [],
    'Percentage': []
}

# Calculate total composition counts for percentages
a_comp_total = sum(a_comp_counter.values())
b_comp_total = sum(b_comp_counter.values())
x_comp_total = sum(x_comp_counter.values())

# Fill the composition data
for site, counter, total in [('A-site', a_comp_counter, a_comp_total), 
                           ('B-site', b_comp_counter, b_comp_total), 
                           ('X-site', x_comp_counter, x_comp_total)]:
    for composition, count in counter.most_common():
        composition_data['Site'].append(site)
        composition_data['Composition'].append(composition)
        composition_data['Count'].append(count)
        composition_data['Type'].append('Double' if len(composition) > 2 else 'Single')
        composition_data['Percentage'].append(round(count/total*100, 2) if total > 0 else 0)

composition_df = pd.DataFrame(composition_data)

# Save to CSV files
element_df.to_csv(f'{results_dir}/element_distribution_{timestamp}.csv', index=False)
composition_df.to_csv(f'{results_dir}/composition_distribution_{timestamp}.csv', index=False)

# Create summary tables
top_elements_per_site = element_df.groupby('Site').apply(lambda x: x.nlargest(10, 'Count')).reset_index(drop=True)
top_compositions_per_site = composition_df.groupby('Site').apply(lambda x: x.nlargest(10, 'Count')).reset_index(drop=True)

top_elements_per_site.to_csv(f'{results_dir}/top_elements_per_site_{timestamp}.csv', index=False)
top_compositions_per_site.to_csv(f'{results_dir}/top_compositions_per_site_{timestamp}.csv', index=False)

# Create pivot tables
pivot_table_elements = element_df.pivot(index='Element', columns='Site', values='Count').fillna(0)
pivot_table_compositions = composition_df.pivot(index='Composition', columns='Site', values='Count').fillna(0)

pivot_table_elements.to_csv(f'{results_dir}/element_pivot_table_{timestamp}.csv')
pivot_table_compositions.to_csv(f'{results_dir}/composition_pivot_table_{timestamp}.csv')

# Plotting function to ensure consistent style
def setup_plot_style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12

setup_plot_style()

# Plot 1: Element distribution bar charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, site in enumerate(['A-site', 'B-site', 'X-site']):
    site_data = element_df[element_df['Site'] == site].nlargest(10, 'Count')
    ax = axes[idx]
    bars = ax.bar(site_data['Element'], site_data['Count'], color=plt.cm.viridis(idx/2))
    ax.set_title(f'Top 10 {site} Elements')
    ax.set_xlabel('Element')
    ax.set_ylabel('Count')
    ax.set_xticklabels(site_data['Element'], rotation=45, ha='right')
    
    # Add percentage labels on bars
    for i, (count, percentage) in enumerate(zip(site_data['Count'], site_data['Percentage'])):
        ax.text(i, count + 0.5, f'{percentage}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{results_dir}/element_distribution_bars_{timestamp}.png', bbox_inches='tight')
plt.close()

# Plot 2: Composition distribution bar charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, site in enumerate(['A-site', 'B-site', 'X-site']):
    site_data = composition_df[composition_df['Site'] == site].nlargest(10, 'Count')
    ax = axes[idx]
    
    # Color-code single vs double compositions
    colors = ['#1f77b4' if t == 'Single' else '#ff7f0e' for t in site_data['Type']]
    bars = ax.bar(site_data['Composition'], site_data['Count'], color=colors)
    
    ax.set_title(f'Top 10 {site} Compositions')
    ax.set_xlabel('Composition')
    ax.set_ylabel('Count')
    ax.set_xticklabels(site_data['Composition'], rotation=45, ha='right')
    
    # Add percentage labels on bars
    for i, (count, percentage) in enumerate(zip(site_data['Count'], site_data['Percentage'])):
        ax.text(i, count + 0.5, f'{percentage}%', ha='center', va='bottom')
    
    # Add legend for single/double
    if idx == 0:  # Only add legend to first plot
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Single'),
                          Patch(facecolor='#ff7f0e', label='Double')]
        ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(f'{results_dir}/composition_distribution_bars_{timestamp}.png', bbox_inches='tight')
plt.close()

# Plot 3: Pie charts for composition types
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, site in enumerate(['A-site', 'B-site', 'X-site']):
    site_data = composition_df[composition_df['Site'] == site]
    type_counts = site_data.groupby('Type')['Count'].sum()
    
    ax = axes[idx]
    wedges, texts, autotexts = ax.pie(type_counts, 
                                     labels=type_counts.index, 
                                     autopct='%1.1f%%',
                                     colors=['#1f77b4', '#ff7f0e'],
                                     startangle=90)
    
    # Make text more readable
    for text in texts + autotexts:
        text.set_fontsize(12)
    
    ax.set_title(f'{site} Composition Types')

plt.tight_layout()
plt.savefig(f'{results_dir}/composition_type_pies_{timestamp}.png', bbox_inches='tight')
plt.close()

# Plot 4: Heatmap of element distribution
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table_elements, annot=True, fmt='g', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('Element Distribution Across Sites')
plt.tight_layout()
plt.savefig(f'{results_dir}/element_distribution_heatmap_{timestamp}.png', bbox_inches='tight')
plt.close()

# Plot 5: Heatmap of composition distribution
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table_compositions, annot=True, fmt='g', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('Composition Distribution Across Sites')
plt.tight_layout()
plt.savefig(f'{results_dir}/composition_distribution_heatmap_{timestamp}.png', bbox_inches='tight')
plt.close()

# Plot 6: Stacked bar chart showing proportion of single vs double compositions
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

site_composition_types = composition_df.groupby(['Site', 'Type'])['Count'].sum().unstack()
site_composition_types = site_composition_types.div(site_composition_types.sum(axis=1), axis=0) * 100

site_composition_types.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
ax.set_ylabel('Percentage')
ax.set_title('Single vs Double Composition Distribution by Site')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Composition Type')

# Add percentage labels
for i, site in enumerate(site_composition_types.index):
    cumulative = 0
    for j, comp_type in enumerate(site_composition_types.columns):
        value = site_composition_types.loc[site, comp_type]
        if not pd.isna(value):
            ax.text(i, cumulative + value/2, f'{value:.1f}%', ha='center', va='center')
            cumulative += value

plt.tight_layout()
plt.savefig(f'{results_dir}/composition_type_stacked_{timestamp}.png', bbox_inches='tight')
plt.close()

# Create a summary report
with open(f'{results_dir}/analysis_report_{timestamp}.txt', 'w') as f:
    f.write("Perovskite Composition Analysis Report\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total number of unique compounds analyzed: {len(names)}\n\n")
    
    f.write("Element Distribution Summary:\n")
    f.write("-" * 30 + "\n")
    for site in ['A-site', 'B-site', 'X-site']:
        site_data = element_df[element_df['Site'] == site].nlargest(5, 'Count')
        f.write(f"\nTop 5 {site} elements:\n")
        for _, row in site_data.iterrows():
            f.write(f"  {row['Element']}: {row['Count']} ({row['Percentage']}%)\n")
    
    f.write("\n\nComposition Distribution Summary:\n")
    f.write("-" * 30 + "\n")
    for site in ['A-site', 'B-site', 'X-site']:
        site_data = composition_df[composition_df['Site'] == site].nlargest(5, 'Count')
        f.write(f"\nTop 5 {site} compositions:\n")
        for _, row in site_data.iterrows():
            f.write(f"  {row['Composition']} ({row['Type']}): {row['Count']} ({row['Percentage']}%)\n")
    
    f.write("\n\nComposition Type Summary:\n")
    f.write("-" * 30 + "\n")
    for site in ['A-site', 'B-site', 'X-site']:
        site_data = composition_df[composition_df['Site'] == site]
        type_counts = site_data.groupby('Type')['Count'].sum()
        total = type_counts.sum()
        f.write(f"\n{site}:\n")
        for comp_type, count in type_counts.items():
            percentage = count/total*100 if total > 0 else 0
            f.write(f"  {comp_type}: {count} ({percentage:.1f}%)\n")

print(f"Analysis completed. Results saved to {results_dir}/")
print(f"Generated files:")
print(f"  - element_distribution_{timestamp}.csv")
print(f"  - composition_distribution_{timestamp}.csv")
print(f"  - top_elements_per_site_{timestamp}.csv")
print(f"  - top_compositions_per_site_{timestamp}.csv")
print(f"  - element_pivot_table_{timestamp}.csv")
print(f"  - composition_pivot_table_{timestamp}.csv")
print(f"  - Various plots (PNG format)")
print(f"  - analysis_report_{timestamp}.txt")