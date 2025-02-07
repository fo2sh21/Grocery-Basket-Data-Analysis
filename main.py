
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from apyori import apriori
import plotly.express as px
import plotly.graph_objects as go
items = []
customers = []
cities = []

def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
        df = pd.DataFrame(data)
        return df
def clean_data(data):
    data = data.dropna()
    df = data[data['count'] < 25]
    df = df.drop_duplicates()
    return df


def extract(df):
    global items, cities, customers

    new_items = df['items'].str.split(',').explode().unique()
    for item in new_items:
        if item not in items:
            items.append(item)

    new_cities = df['city'].unique()
    for city in new_cities:
        if city not in cities:
            cities.append(city)

    grouped = df.groupby('customer')
    for customer, group in grouped:
        customer_cities = group['city'].unique().tolist()
        customer_age = group['age'].iloc[0]
        cash_total = group[group['paymentType'] == 'Cash']['total'].sum()
        credit = group[group['paymentType'] == 'Credit']['total'].sum()
        payment = [int(cash_total), int(credit)]
        customers.append({
            'customer': customer,
            'cities': customer_cities,
            'age': int(customer_age),
            'payment': payment
        })

def kmeans(n_clusters):
    customer_data = np.array([[c['age'], c['payment'][0] + c['payment'][1]] for c in customers])
    customer_names = [c['customer'] for c in customers]
    kmeans = KMeans(n_clusters=n_clusters, random_state=23)
    cluster_labels = kmeans.fit_predict(customer_data)

    # Create descriptive cluster labels
    cluster_labels_descriptive = [f'Cluster {i+1}' for i in cluster_labels]

    df = pd.DataFrame({
        'Customer': customer_names,
        'Age': customer_data[:, 0],
        'Total Spending': customer_data[:, 1],
        'Cluster': cluster_labels_descriptive
    })

    fig = px.scatter(df, x='Age', y='Total Spending', color='Cluster', title='K-Means Clustering', hover_data=['Customer'])
    fig.update_layout(paper_bgcolor='#F0F2F6', plot_bgcolor='#FFFFFF', title_font_size=20, title_x=0.5, xaxis_title_font_size=15, yaxis_title_font_size=15)

    return fig
#def AR(df):
    #transactions = df['items']
    #transactions = transactions.str.split(',')
    #transactions = transactions.to_list()
    #num_transactions = len(transactions)
    #support_threshold = int(num_transactions * 0.01)  # 1% of total transactions
    #patterns = pyfpgrowth.find_frequent_patterns(transactions, support_threshold)
    #rules = pyfpgrowth.generate_association_rules(patterns, 0.1)
    #return rules
def AR_apriori(df,min_support=0.01, min_confidence=0.1):
    transactions = df['items']
    transactions = transactions.str.split(',')
    transactions = transactions.to_list()
    rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=1, min_length=2)
    results = list(rules)
    return results


def payment_type_graph():
    cash = 0
    credit = 0
    for customer in customers:
        cash += customer['payment'][0]
        credit += customer['payment'][1]
    payments = [cash, credit]
    labels = ['Cash', 'Credit']

    fig = px.pie(values=payments, names=labels, title='Payment Type Distribution', hole=0.3)
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0], marker=dict(colors=['#FF4B4B', '#31333F']))
    fig.update_layout(title_font_size=20, title_x=0.5, legend_title_text='Payment Types', paper_bgcolor='#F0F2F6', plot_bgcolor='#FFFFFF')
    return fig

def spending_by_age():
    ages = np.unique([c['age'] for c in customers])
    total = []
    for age in ages:
        spent = 0
        for customer in customers:
            if customer['age'] == age:
                spent += customer['payment'][0] + customer['payment'][1]
        total.append({'age': int(age), 'total': spent})

    df = pd.DataFrame(total)
    fig = px.scatter(df, x='age', y='total', title='Total Spending by Age', labels={'age': 'Age', 'total': 'Total Spending'})
    fig.update_traces(mode='lines+markers', marker=dict(size=10, color='#FF4B4B', line=dict(width=2, color='#31333F')))
    fig.update_layout(title_font_size=20, title_x=0.5, xaxis_title_font_size=15, yaxis_title_font_size=15, paper_bgcolor='#F0F2F6', plot_bgcolor='#FFFFFF')
    return fig

def spending_by_city():
    city_totals = []
    for city in cities:
        spent = 0
        for customer in customers:
            if city in customer['cities']:
                spent += customer['payment'][0] + customer['payment'][1]
        city_totals.append({'city': city, 'total': spent})
    city_totals = sorted(city_totals, key=lambda x: x['total'], reverse=True)

    df = pd.DataFrame(city_totals)
    fig = px.bar(df, x='city', y='total', title='Total Spending by City', labels={'city': 'City', 'total': 'Total Spending'}, color='city')
    fig.update_traces(marker=dict(line=dict(width=1.5, color='#31333F'), opacity=0.6))
    fig.update_layout(title_font_size=20, title_x=0.5, xaxis_title_font_size=15, yaxis_title_font_size=15, paper_bgcolor='#F0F2F6', plot_bgcolor='#FFFFFF', legend_title_text='Cities')
    return fig

def distibution_graph(df):
    max_spent = df['total'].max()
    least_spent = df['total'].min()
    no_of_bins = int(max_spent / least_spent)

    fig = go.Figure(data=[go.Histogram(x=df['total'], nbinsx=no_of_bins, marker_color='#FF4B4B', opacity=0.75)])
    fig.update_layout(title='Total Distribution of Spending', xaxis_title='Total Spending', yaxis_title='Frequency', title_font_size=20, title_x=0.5, xaxis_title_font_size=15, yaxis_title_font_size=15, paper_bgcolor='#F0F2F6', plot_bgcolor='#FFFFFF', bargap=0.2)
    return fig

def main():
    df = read_csv("C:/Users/fadyw/Documents/R PROJECT/members code/Fo2sh/grc.csv")
    df = clean_data(df)
    extract(df)
    kmeans()
    rules = AR_apriori(df)
    payment_type_graph()
    spending_by_age()
    spending_by_city()
    distibution_graph(df)



if __name__ == '__main__':
    main()

