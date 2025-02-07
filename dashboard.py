import streamlit as st
import pandas as pd
import main
import plotly.express as px

# Remove theme configuration, keep other settings
st.set_page_config(page_title="Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for consistent dataframe styling
st.markdown("""
    <style>
    .dataframe {
        padding: 1rem !important;
        margin: 2rem 0 !important;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    .stat-value {
        font-size: 1.2rem;
        color: #FF4B4B;
        font-weight: bold;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# Define dark theme template for plots
PLOT_TEMPLATE = {
    'layout': {
        'plot_bgcolor': '#1f2937',
        'paper_bgcolor': '#1f2937',
        'font': {'color': '#ffffff'},
        'title': {'font': {'color': '#ffffff'}},
        'xaxis': {'gridcolor': '#374151', 'color': '#ffffff'},
        'yaxis': {'gridcolor': '#374151', 'color': '#ffffff'},
        'legend': {'font': {'color': '#ffffff'}}
    }
}

st.title('Dashboard')

# Define tabs
clean, visualize, cluster, ar, lists, customers = st.tabs(['Data Cleaning', 'Visualization', 'Clustering', 'Association Rules', 'Lists', 'Customers'])

global file

def rules_to_dataframe(rules):
    rules_data = []
    for rule in rules:
        for ordered_stat in rule.ordered_statistics:
            rules_data.append({
                'Antecedents': ', '.join(list(ordered_stat.items_base)),
                'Consequents': ', '.join(list(ordered_stat.items_add)),
                'Support': rule.support,
                'Confidence': ordered_stat.confidence,
                'Lift': ordered_stat.lift
            })
    dataframe = pd.DataFrame(rules_data)
    return dataframe

with clean:
    file = st.file_uploader('Upload a file')
    if file is not None:
        df = pd.read_csv(file)
        df = main.clean_data(df)
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.dataframe(df, height=600, width=1920)
        main.extract(df)

with visualize:
    if file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Payment Type Graph")
            payment_fig = main.payment_type_graph()
            payment_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font=dict(color='white')
            )
            st.plotly_chart(payment_fig)
            
            st.write("Spending by City")
            city_fig = main.spending_by_city()
            city_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font=dict(color='white')
            )
            st.plotly_chart(city_fig)

        with col2:
            st.write("Spending by Age")
            age_fig = main.spending_by_age()
            age_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font=dict(color='white')
            )
            st.plotly_chart(age_fig)
            
            st.write("Distribution Graph")
            dist_fig = main.distibution_graph(df)
            dist_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font=dict(color='white')
            )
            st.plotly_chart(dist_fig)

with cluster:
    if file is not None:
        st.write("K-Means Clustering")
        k = st.slider("Select centers 'K's'", 2, 4, 2)
        fig = main.kmeans(k)
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font=dict(color='white')
        )
        st.plotly_chart(fig)

with ar:
    support = st.slider("support", 0.01, 1.0, 0.01)
    confidence = st.slider("confidence", 0.01, 1.0, 0.01)
    if file is not None:
        st.write("Association Rules")
        rules = main.AR_apriori(df, support, confidence)
        dataframe = rules_to_dataframe(rules)
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.dataframe(dataframe, height=600, width=1920)

with lists:
    if file is not None:
        subtab1, subtab2, subtab3 = st.tabs(['Items', 'Cities', 'Payment Type'])

        with subtab1:
            selected_items = st.multiselect('Select Items', main.items)
            filtered_df = df.copy()
            if selected_items:
                filtered_df = filtered_df[filtered_df['items'].apply(lambda x: any(item in x for item in selected_items))]
            st.dataframe(
                filtered_df,
                height=400,
                use_container_width=True
            )

        with subtab2:
            selected_cities = st.multiselect('Select Cities', main.cities)
            filtered_df = df.copy()
            if selected_cities:
                filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]
            st.dataframe(
                filtered_df,
                height=400,
                use_container_width=True
            )

        with subtab3:
            selected_payment_type = st.selectbox('Select Payment Type', ['Cash', 'Credit'])
            filtered_df = df.copy()
            if selected_payment_type:
                filtered_df = filtered_df[filtered_df['paymentType'] == selected_payment_type]
            st.dataframe(
                filtered_df,
                height=400,
                use_container_width=True
            )

with customers:
    if file is not None:
        # Initialize session state
        if 'customer_data' not in st.session_state:
            st.session_state.customer_data = {
                'selected_index': 0,
                'previous_selection': None
            }
        
        # Header section
        st.title('Customer Analysis')
        customer_names = tuple(c['customer'] for c in main.customers)
        
        def on_customer_select():
            st.session_state.customer_data['selected_index'] = customer_names.index(st.session_state.customer_select)
            st.session_state.customer_data['previous_selection'] = st.session_state.customer_select
        
        selected_customer = st.selectbox(
            'Select a customer to view their details',
            options=customer_names,
            index=st.session_state.customer_data['selected_index'],
            key='customer_select',
            on_change=on_customer_select
        )

        if selected_customer:
            customer_data = next(c for c in main.customers if c['customer'] == selected_customer)
            
            # Customer Details Box
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Customer Details")
                st.markdown(f"""
                    **Customer ID:** <span class="stat-value">{customer_data['customer']}</span><br>
                    **Age:** <span class="stat-value">{customer_data['age']}</span><br>
                    **Cities Visited:** <span class="stat-value">{', '.join(customer_data['cities'])}</span><br>
                    **Total Spending:** <span class="stat-value">${sum(customer_data['payment']):.2f}</span>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Payment Distribution")
                labels = ['Cash', 'Credit']
                values = customer_data['payment']
                fig = px.pie(
                    values=values, 
                    names=labels, 
                    title='',
                    hole=0.3,
                    template='plotly_dark'
                )
                fig.update_traces(
                    textinfo='percent+label', 
                    pull=[0.1, 0], 
                    marker=dict(colors=['#FF4B4B', '#31333F'])
                )
                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(t=0, l=0, r=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

            # Orders Box
            st.subheader("Customer Orders")
            customer_orders = df[df['customer'] == selected_customer]
            
            # Add summary metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Orders", len(customer_orders), delta=None)
            with col2:
                st.metric("Average Order Value", f"${customer_orders['total'].mean():.2f}", delta=None)
            
            # Orders table
            st.dataframe(
                customer_orders,
                height=400,
                use_container_width=True
            )

#start the app
