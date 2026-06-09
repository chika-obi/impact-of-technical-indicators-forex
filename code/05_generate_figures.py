"""
Step 5: Generate all figures for the paper
Includes: accuracy comparison bar chart, degradation plot, feature importance bar chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("=" * 60)
print("STEP 5: Generating Figures for Paper")
print("=" * 60)

# ============================================================================
# LOAD RESULTS
# ============================================================================

# Load ablation results from Step 3
results_df = pd.read_csv('results/baseline_vs_full_results.csv')

# Extract numeric values
results_df['Baseline'] = results_df['Baseline_Accuracy'].str.replace('%', '').astype(float)
results_df['Full_Set'] = results_df['Full_Set_Accuracy'].str.replace('%', '').astype(float)
results_df['Diff'] = results_df['Difference'].str.replace('pp', '').astype(float)

print(f"\nLoaded {len(results_df)} currency pairs from results/baseline_vs_full_results.csv")

# Load feature importance from Step 4
try:
    importance_df = pd.read_csv('results/feature_importance.csv')
    print(f"Loaded feature importance from results/feature_importance.csv")
except:
    print("Warning: feature_importance.csv not found. Run Step 4 first.")
    importance_df = None

# ============================================================================
# FIGURE 1: Baseline vs Full Set Accuracy (Top 10 by degradation)
# ============================================================================
print("\n📊 Generating Figure 1: Accuracy Comparison...")

top_10 = results_df.nlargest(10, 'Diff').copy()
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(top_10))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], top_10['Baseline'], width, 
               label='Baseline (OHLC only)', color='#2E86AB')
bars2 = ax.bar([i + width/2 for i in x], top_10['Full_Set'], width, 
               label='Full Set (OHLC + 11 Indicators)', color='#A23B72')

ax.set_xlabel('Currency Pair', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Figure 1: Baseline vs Full Set Accuracy\n(Top 10 Pairs by Degradation)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(top_10['Currency'], rotation=45, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(40, 75)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figures/figure1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: figures/figure1_accuracy_comparison.png")

# ============================================================================
# FIGURE 2: Degradation by Currency Pair (All 27 pairs)
# ============================================================================
print("\n📊 Generating Figure 2: Degradation by Currency Pair...")

fig, ax = plt.subplots(figsize=(12, 10))
degradation_df = results_df.sort_values('Diff', ascending=True)
colors = ['#d73027' if d < 0 else '#1a9850' for d in degradation_df['Diff']]
bars = ax.barh(degradation_df['Currency'], degradation_df['Diff'], color=colors)

ax.axvline(x=0, color='black', linewidth=1.5)
ax.set_xlabel('Change in Accuracy (percentage points)', fontsize=12)
ax.set_title('Figure 2: Impact of Adding Technical Indicators\n(Baseline vs Full Set Across All 27 Pairs)', fontsize=14)
ax.set_xlim(-13, 2)

# Add value labels
for bar, diff in zip(bars, degradation_df['Diff']):
    if diff < -5:
        ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2,
                f'{diff:.1f}pp', ha='right', va='center', fontsize=8, color='white')
    else:
        ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2,
                f'{diff:.1f}pp', ha='right', va='center', fontsize=8, color='black')

plt.tight_layout()
plt.savefig('figures/figure2_degradation_by_pair.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: figures/figure2_degradation_by_pair.png")

# ============================================================================
# FIGURE 3: Feature Importance Bar Chart
# ============================================================================
if importance_df is not None:
    print("\n📊 Generating Figure 3: Feature Importance...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color coding: Blue for OHLC, Purple for indicators
    ohlc_features = ['Open', 'High', 'Low', 'Close']
    colors = ['#2E86AB' if f in ohlc_features else '#A23B72' for f in importance_df['Feature']]
    
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Figure 3: Random Forest Feature Importance Rankings', fontsize=14)
    ax.axvline(x=importance_df['Importance'].mean(), color='gray', linestyle='--', 
               label=f'Mean: {importance_df["Importance"].mean():.3f}')
    
    # Add value labels
    for bar, imp in zip(bars, importance_df['Importance']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E86AB', label='OHLC Features (4)'),
                       Patch(facecolor='#A23B72', label='Technical Indicators (11)')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('figures/figure3_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: figures/figure3_feature_importance.png")
else:
    print("\n⚠️ Skipping Figure 3: feature_importance.csv not found")

# ============================================================================
# FIGURE 4: Summary Statistics Dashboard (Simple)
# ============================================================================
print("\n📊 Generating Figure 4: Summary Statistics...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Calculate statistics
baseline_wins = (results_df['Diff'] < 0).sum()
full_wins = (results_df['Diff'] > 0).sum()
avg_baseline = results_df['Baseline'].mean()
avg_full = results_df['Full_Set'].mean()
avg_degradation = results_df['Diff'].mean()

# Create summary text
summary_text = f"""
SUMMARY STATISTICS (27 Currency Pairs)
======================================

Baseline (OHLC only) wins:     {baseline_wins} / 27 pairs
Full Set wins:                 {full_wins} / 27 pairs

Average Baseline Accuracy:     {avg_baseline:.2f}%
Average Full Set Accuracy:     {avg_full:.2f}%
Average Degradation:           {avg_degradation:+.2f} percentage points

Worst Degradation:             {results_df['Diff'].min():+.2f}pp ({results_df.loc[results_df['Diff'].idxmin(), 'Currency']})
Smallest Degradation:          {results_df['Diff'].max():+.2f}pp ({results_df.loc[results_df['Diff'].idxmax(), 'Currency']})

Highest Baseline Accuracy:     {results_df['Baseline'].max():.2f}% ({results_df.loc[results_df['Baseline'].idxmax(), 'Currency']})
Lowest Baseline Accuracy:      {results_df['Baseline'].min():.2f}% ({results_df.loc[results_df['Baseline'].idxmin(), 'Currency']})
"""

ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12, 
        verticalalignment='center', fontfamily='monospace')

plt.title('Figure 4: Summary of Results', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('figures/figure4_summary_statistics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: figures/figure4_summary_statistics.png")

# ============================================================================
# FIGURE 5: Category Comparison (Major vs Minor vs Exotic)
# ============================================================================
print("\n📊 Generating Figure 5: Category Comparison...")

# Define categories
major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD']
minor_pairs = ['EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/CAD', 'GBP/JPY', 'GBP/CHF', 'GBP/AUD', 'GBP/CAD', 'AUD/JPY']
exotic_pairs = ['USD/ZAR', 'USD/SEK', 'USD/NOK', 'USD/MXN', 'USD/TRY', 'USD/SGD', 'USD/HKD', 'USD/HUF', 'USD/PLN', 'USD/CZK']

# Calculate averages by category
results_df['Category'] = 'Other'
results_df.loc[results_df['Currency'].isin(major_pairs), 'Category'] = 'Major'
results_df.loc[results_df['Currency'].isin(minor_pairs), 'Category'] = 'Minor'
results_df.loc[results_df['Currency'].isin(exotic_pairs), 'Category'] = 'Exotic'

category_stats = results_df.groupby('Category').agg({
    'Baseline': 'mean',
    'Full_Set': 'mean',
    'Diff': 'mean'
}).round(2)

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(category_stats))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], category_stats['Baseline'], width, 
               label='Baseline (OHLC only)', color='#2E86AB')
bars2 = ax.bar([i + width/2 for i in x], category_stats['Full_Set'], width, 
               label='Full Set (OHLC + 11 Indicators)', color='#A23B72')

ax.set_xlabel('Currency Category', fontsize=12)
ax.set_ylabel('Average Accuracy (%)', fontsize=12)
ax.set_title('Figure 5: Performance by Currency Category', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(category_stats.index)
ax.legend()

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/figure5_category_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: figures/figure5_category_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\nFigures saved in /figures directory:")
print("  📊 figure1_accuracy_comparison.png  - Baseline vs Full Set (Top 10)")
print("  📊 figure2_degradation_by_pair.png  - Degradation across all 27 pairs")
print("  📊 figure3_feature_importance.png   - Feature importance rankings")
print("  📊 figure4_summary_statistics.png   - Summary statistics dashboard")
print("  📊 figure5_category_comparison.png  - Major vs Minor vs Exotic")

print("\n" + "=" * 60)
print("✅ Step 5 Complete!")
print("=" * 60)