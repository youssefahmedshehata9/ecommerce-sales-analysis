import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px 

# قراءة البيانات
df = pd.read_csv("online_retail_II.csv")
df.columns = df.columns.str.strip()

# نظرة أولية
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

# تنظيف البيانات
df_clean = df.copy()
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.dropna(subset=['Customer ID'])
df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
df_clean = df_clean.dropna(subset=['Description'])
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
df_clean['Total'] = df_clean['Quantity'] * df_clean['Price']

print(df_clean.shape)
print(df_clean.isnull().sum())

# إنشاء عمود شهر وسنة
df_clean['Month'] = df_clean['InvoiceDate'].dt.to_period('M')

# الإيرادات الشهرية
monthly_revenue = df_clean.groupby('Month')['Total'].sum().reset_index()
monthly_revenue['Month'] = monthly_revenue['Month'].dt.to_timestamp()

# رسم الإيرادات الشهرية
plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_revenue, x='Month', y='Total', marker='o')
plt.title('Monthly Revenue Over Time', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Revenue (£)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# باستخدام Plotly
fig = px.line(monthly_revenue, x='Month', y='Total',
              title='Monthly Revenue Over Time',
              labels={'Total': 'Revenue (£)', 'Month': 'Month'},
              markers=True)
fig.show()

# الإيرادات حسب الدولة (بدون UK)
country_revenue = df_clean.groupby('Country')['Total'].sum().sort_values(ascending=False).reset_index()
top_countries = country_revenue[country_revenue['Country'] != 'United Kingdom'].head(10)

plt.figure(figsize=(12,6))
sns.barplot(data=top_countries, x='Total', y='Country', palette='viridis')
plt.title('Top 10 Countries by Revenue (Excl. UK)', fontsize=16)
plt.xlabel('Revenue (£)')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

# أكثر المنتجات مبيعًا
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=top_products, x='Quantity', y='Description', palette='magma')
plt.title('Top 10 Best-Selling Products by Quantity', fontsize=16)
plt.xlabel('Quantity Sold')
plt.ylabel('Product')
plt.tight_layout()
plt.show()

# أعلى العملاء إنفاقًا
top_customers = df_clean.groupby('Customer ID')['Total'].sum().sort_values(ascending=False).head(10).reset_index()
top_customers['Customer ID'] = top_customers['Customer ID'].astype(int).astype(str)

plt.figure(figsize=(10,5))
sns.barplot(data=top_customers, x='Customer ID', y='Total', palette='coolwarm')
plt.title('Top 10 Customers by Revenue', fontsize=16)
plt.xlabel('Customer ID')
plt.ylabel('Total Spend (£)')
plt.tight_layout()
for i, row in top_customers.iterrows():
    plt.text(i, row['Total'] + 50, f"£{row['Total']:.0f}", ha='center', fontsize=9)
plt.show()

# 🔥 Heatmap للمبيعات حسب اليوم 🔥
# استخراج يوم الأسبوع
df_clean['Day'] = df_clean['InvoiceDate'].dt.day_name()

# ترتيب الأيام
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# تجميع الإيرادات حسب اليوم
daily_revenue = df_clean.groupby('Day')['Total'].sum().reindex(order).reset_index()

# تحويل البيانات لهيكل heatmap
heatmap_data = daily_revenue.pivot_table(index='Day', values='Total')

# رسم heatmap
plt.figure(figsize=(6,5))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title('Total Revenue by Day of the Week', fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()
