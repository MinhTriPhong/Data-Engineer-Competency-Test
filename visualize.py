import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('Competency_Test_Contracts_20250721_rf_predicted.csv')

df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])

fig = plt.figure(figsize=(22, 18))

plt.subplots_adjust(hspace=0.6, wspace=0.6)

ax1 = plt.subplot(2, 3, 1)
contract_types = df.groupby('Contract Type')['Award Amount'].sum()
ax1.pie(contract_types, autopct='%1.1f%%', labels=None)
ax1.set_title('Contract Distribution by Type', fontsize=14)

ax2 = plt.subplot(2, 3, 2)
dept_spending = df.groupby('Department')['Award Amount'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=dept_spending.values, y=dept_spending.index, ax=ax2)
ax2.set_title('Top 10 Departments by Spending', fontsize=14)
ax2.set_xlabel('Award Amount')

ax3 = plt.subplot(2, 3, 3)
vendor_counts = df.groupby('Department')['Vendor Name'].nunique().sort_values(ascending=False).head(10)
sns.barplot(x=vendor_counts.values, y=vendor_counts.index, ax=ax3)
ax3.set_title('Top 10 Departments by Vendor Count', fontsize=14)
ax3.set_xlabel('Number of Unique Vendors')

ax5 = plt.subplot(2, 3, 5)
df['Year'] = df['End Date'].dt.year
df['Month'] = df['End Date'].dt.month
heatmap_data = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
heatmap_pivot = heatmap_data.pivot(index='Year', columns='Month', values='Count').fillna(0)
sns.heatmap(heatmap_pivot, cmap="YlOrRd", ax=ax5)
ax5.set_title('Contract Renewal Calendar', fontsize=14)
ax5.set_xlabel('Month')
ax5.set_ylabel('Year')

plt.suptitle('Contract Management Dashboard', fontsize=24)

plt.savefig('contract_dashboard.png', bbox_inches='tight', dpi=300)

plt.show()