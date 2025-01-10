import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns


# Download necessary images 
logo = Image.open("file.png")

#load Data for vizualizations 
df=pd.read_csv("ABCDE_final.csv")

#Creating list of features 
metric_features_viz  = ["Monetary_Spending","Recency","Weekdays_Transactions", "Weekends_Transactions", "Morning_Transactions", 
                                "Afternoon_Transactions", "Evening_Transactions","Night_Transactions","Engagement_Span", 'CUI_American', 
                                'CUI_Healthy', 'CUI_Indian', 'CUI_Italian', 'CUI_OTHER', 'CUI_Asian_Fusion', 
                                'CUI_Sweets_and_Beverages', 'CUI_Snacks']

metric_features_final =['vendor_count','first_order','CUI_American','CUI_Healthy','CUI_Indian','CUI_Italian','CUI_OTHER','Weekdays_Transactions',
    'Weekends_Transactions','Morning_Transactions','Afternoon_Transactions','Evening_Transactions','Night_Transactions','Monetary_Spending',
    'Engagement_Span','Recency','CUI_Asian_Fusion','CUI_Sweets_and_Beverages','CUI_Snacks']

profiling_values = ['customer_region_2360', 'customer_region_2440', 'customer_region_2490',
       'customer_region_4140', 'customer_region_4660', 'customer_region_8370',
       'customer_region_8550', 'customer_region_8670', 'is_chain_0',
       'is_chain_1', 'last_promo_-', 'last_promo_DELIVERY',
       'last_promo_DISCOUNT', 'last_promo_FREEBIE', 'payment_method_CARD',
       'payment_method_CASH', 'payment_method_DIGI',
       'Buyer_type_OneTime_Buyer', 'Buyer_type_Repeated_Buyer',
       'Spending_Budget_Q1 (Low)', 'Spending_Budget_Q2 (Medium-Low)',
       'Spending_Budget_Q3 (Medium-High)', 'Spending_Budget_Q4 (High)',
       'Consumer_Segment_Group/Family', 'Consumer_Segment_Individual',
       'cities_2', 'cities_4', 'cities_8']

preference_labels = [
    'CUI_American', 'CUI_Healthy', 'CUI_Indian', 'CUI_Italian',
    'CUI_OTHER', 'CUI_Asian_Fusion', 'CUI_Sweets_and_Beverages', 'CUI_Snacks'
]

all_metric_features = (metric_features_final +['merged_labels'] + ['preference_labels'] + ['behavior_labels'])

col_1, col_2, col_3 = st.columns(3)
bottom_left, bottom_right = st.columns(2)
#st.set_page_config(layout='wide')


with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Dashboard"], 
        icons=["house", "bar-chart-line"], menu_icon="cast", default_index=1)
    selected

st.logo(logo, size="large")

# Home Page
if selected == "Home":
    st.title("Customer Segmentation Dashboard App")

    # Dropdown for team members
    st.subheader("Project Team")
    with st.expander("Click to view team details"):
        team_members = {
            "Data Analyst ": ("Ana Caleiro", "20240696"),
            "Data Scientist": ("Érica Parracho", "20240583"),
            "Data Scientist ": ("Oumaima Hfaiedh", "20240699"),
            "Data Analyst ": ("Rute Teixeira", "20240667"),
        }
        for role, (name, student_id) in team_members.items():
            st.write(f"**{role}**: {name} ({student_id})")

    # App description
    st.markdown("""
    This dashboard is designed to help ABCDEats visualize and analyze customer segmentation results, supporting better marketing strategies to improve customer retention and engagement. Using data collected over three months, clustering models like K-means, Hierarchical Clustering, Self-Organizing Maps (SOMs), were applied to group customers based on behaviors and preferences.
    
    The dashboard provides:
    
    Cluster Exlporation: Interactive and static plots to explore the unique traits of each customer group wich provides actionable insights to guide personalized marketing and engagement strategies.
    
    With this tool, ABCDEats can turn complex data into clear insight This dashboard bridges the gap between advanced analytics and actionable business insights, empowering ABCDEats to foster stronger customer connections.
     """)

    #Inputing the metadata 
    st.subheader("Metadata")
    with st.expander("Click to view detailed information about the questions in the form"):
        metadata = {
                "customer_id": "Unique identifier for each customer.",
    "customer_region": "Geographic region where the customer is located.",
    "customer_age": "Age of the customer.",
    "vendor_count": "Number of unique vendors the customer has ordered from.",
    "product_count": "Total number of products the customer has ordered.",
    "is_chain": "Indicates whether the customer's order was from a chain restaurant.",
    "first_order": "Number of days from the start of the dataset when the customer first placed an order.",
    "last_order": "Number of days from the start of the dataset when the customer most recently placed an order.",
    "last_promo": "The category of the promotion or discount most recently used by the customer.",
    "payment_method": "Method most recently used by the customer to pay for their orders.",
    "cuisine_spending": "The amount in monetary units spent by the customer on each cuisine.",
    'Recency' : "Calculated by subtracting the last_order value from 90 which represents the last day of the study period.",
    "Engagement Span": "Calculated by computing the difference between last_order and first_order.",
    "Monetary Spending": "Total monetary value for each customer.",
    "Frequency": "Total number of orders for each customer.",
    "Buyer type": "Categorical feature indicating whether a customer is a one-time or repeated buyer.",
    "Spending Budget": "Categorical feature that profiles our customer based on how much they are willing to spend on the business.",
    "Customer Segment":" Created to profile whether transactions were made for an individual purchase or a group/family.",
    "Cities": "Based on our EDA, we observed that regions sharing similar prefixes tend to exhibit the same preferences.",
    "Weekdays and Weekends Transactions":"Captures the behavior of the customers during this two time periods.",
    "Morning Transactions, Afternoon Transactions, Evening Transactions, and Night Transactions" : "Captures the behavior of the customers during this time periods"

    }

        # Transform the Dict in a pandas data frame
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Attribute", "Description"])
        
        # Display the table
        st.dataframe(metadata_df)

    st.markdown("""
                
                We recommend that the client review the insights provided in the report at the following link:https://drive.google.com/file/d/1IYqJXajg1I2jQaD2EMC2C_BEgFWSbXL7/view , so they can make the most of this dashboard.
                
                
                """)

if selected == "Dashboard":
      
    def show_dash1():
        with col_1:
            st.title('Pairwise relationship of Numerical Features')
            def interactive_scater (dataframe):
                x_axis_val = st.selectbox('Select X-Axis Value', options=metric_features_final)
                y_axis_val = st.selectbox('Select Y-Axis Value', options=metric_features_final)
                col = st.color_picker('Select a plot colour', '#1f77b4')

                plot  = px.scatter(dataframe, x=x_axis_val, y=y_axis_val)
                plot.update_traces(marker = dict(color=col))
                st.plotly_chart(plot)

            interactive_scater (df)

        with col_2:
            st.title('Histogram of Numerical Features')
            def interactive_hist (dataframe):
                box_hist = st.selectbox('Feature', options=metric_features_final)
                color_choice = st.color_picker('Select a plot colour',value= '#1f77b4', key='color')
                bin_count = st.slider('Select number of bins', min_value=5, max_value=100, value=20, step=1)

                hist  = sns.displot(dataframe[box_hist], color=color_choice, bins=bin_count)
        
                plt.title(f"Histogram of {box_hist}")
                st.pyplot(hist)

        
            interactive_hist(df)


        with col_3:
            st.title('Customer Preferences Distribution')
            def pie_chart (dataframe): 
                
                # Calculate the sum of each preference label
                preference_data = df[preference_labels].sum()

                # Convert to a DataFrame for easier manipulation
                preference_df = preference_data.reset_index()
                preference_df.columns = ['Preference', 'Total']

                # Interactive pie chart
                fig = px.pie(
                    preference_df,
                    values='Total',
                    names='Preference',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )   

                # Display the chart in Streamlit
                st.plotly_chart(fig)

            pie_chart(df)

        

        with bottom_left:
            st.title('Cluster Profiling Heatmap')
            # Select clustering column
            cluster_col = st.selectbox("Select Cluster Column", options=['merged_labels', 'preference_labels', 'behavior_labels'])

            # Select features for heatmap
            features = st.multiselect("Select Features for Heatmap", options=df.columns, default=['Monetary_Spending', 'Recency'])

            # Compute mean profiles
            if cluster_col in df.columns and features:
                cluster_profile = df.groupby(cluster_col)[features].mean().T

            # Heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(cluster_profile, center=0, annot=True, fmt=".2f", cmap="Blues", ax=ax)

                ax.set_title(f"Cluster Profiling Heatmap ({cluster_col})", fontsize=16)
                st.pyplot(fig)

        with bottom_right:
            st.title('Cluster Profiles')

            def cluster_profiles_plotly(df, label_columns, metric_features):
        
            # Create a dropdown to select the cluster label column
                selected_label = st.selectbox("Select Cluster Label", options=label_columns)

                if selected_label:
                # Filter the DataFrame to only include the selected cluster label and features
                    drop_cols = [col for col in label_columns if col != selected_label]
                    df_filtered = df.drop(columns=drop_cols)

                # Calculate cluster centroids and counts
                    centroids = df_filtered.groupby(selected_label)[metric_features].mean().reset_index()
                    counts = df_filtered[selected_label].value_counts().reset_index()
                    counts.columns = [selected_label, "counts"]

                # Cluster Means Line Chart
                    fig_means = px.line(
                    centroids.melt(id_vars=[selected_label], var_name="Feature", value_name="Mean Value"),
                    x="Feature",
                    y="Mean Value",
                    color=selected_label,
                    title="Cluster Means",
                    markers=True,
                    labels={selected_label: "Cluster"}
                    )   
                    fig_means.update_layout(
                    xaxis=dict(tickangle=45),
                    legend_title="Cluster",
                    height=500,
                    width=800,
                    )
                    st.plotly_chart(fig_means, use_container_width=True)

                    # Cluster Sizes Bar Chart
                    fig_sizes = px.bar(
                    counts,
                    x=selected_label,
                    y="counts",
                    color=selected_label,
                    text="counts",
                    title="Cluster Sizes",
                    labels={"counts": "Number of Samples", selected_label: "Cluster"},
                    color_discrete_sequence=px.colors.qualitative.Pastel2,
                    )
                    fig_sizes.update_traces(textposition='outside')
                    fig_sizes.update_layout(
                    height=400,
                    width=600,
                    )
                    st.plotly_chart(fig_sizes, use_container_width=True)


            cluster_profiles_plotly(df=df[metric_features_final + ['preference_labels','behavior_labels', 'merged_labels']], label_columns=['preference_labels','behavior_labels', 'merged_labels'], metric_features=metric_features_final)
        
        st.divider()

        def segment(dataframe):
            st.title("Segment Profile Visualization")

                #assuming 'merged_labels' is the cluster column
            cluster_col = 'merged_labels'

            if cluster_col in df.columns:
                    # Filter data for valid clusters and features
                    data = df[metric_features_final + [cluster_col]].dropna()
                    clusters = data[cluster_col].unique()

                # Calculate cluster sizes and percentages
            cluster_sizes = data[cluster_col].value_counts().sort_index()
            total_size = len(data)
            cluster_percentages = (cluster_sizes / total_size) * 100

                # Calculate mean profiles for each cluster
            hc_profile = data.groupby(cluster_col)[metric_features_final].mean()

                # Plot setup
            num_clusters = len(clusters)
            num_cols = 3  # Define number of columns in grid
            num_rows = -(-num_clusters // num_cols)  # Compute rows dynamically

            fig, axes = plt.subplots(
                    num_rows, num_cols, 
                    figsize=(18, num_rows * 5), 
                    constrained_layout=True
                )   
            axes = axes.flatten()  # Flatten for easy iteration

            colors = sns.color_palette("Set2", num_clusters)  # Custom color palette for each cluster

                # Iterate over clusters and plot
            for idx, (cluster, ax) in enumerate(zip(clusters, axes)):
                # Mean profile for the cluster
                    mean_profile = hc_profile.loc[cluster]
                    mean_profile.plot.barh(
                        ax=ax,
                        color=colors[idx],
                        alpha=0.8,
                        edgecolor='black'
                    )   
            
            # Add cluster size and percentage to the title
            cluster_size = cluster_sizes[cluster]
            cluster_percentage = cluster_percentages[cluster]
            ax.set_title(f"Cluster {cluster}: {cluster_size} ({cluster_percentage:.1f}%)", fontsize=12)
            ax.set_xlabel("Mean Value")
            ax.set_ylabel("Features")
            ax.tick_params(axis='y', labelsize=10)

        # Hide unused axes
            for extra_ax in axes[num_clusters:]:
                extra_ax.set_visible(False)

        # Overall plot title
            fig.suptitle("Segment Profile Plot", fontsize=16)

        # Show plot in Streamlit
            st.pyplot(fig)
    
        segment(df)
    
        st.divider()

    # Dont forget to write a reccomendation to use all merged labels
        def interactive_barplots(dataframe):
            st.title('Profiling with Categorical Data')
            st.markdown('We advise to choose merged_labels in the label column for the best analysis')
            sns.set_style('white')
        # Allow user to select the merged_labels column
            merged_labels_col = st.selectbox('Select Label Column', options=all_metric_features)
    
        # Filter out numeric features for bar plotting
            available_features = profiling_values
            features = st.multiselect('Select Features to Plot', options=available_features, default=available_features[:4])

        # Choose the color for the bars
            color_choice = st.color_picker('Select Bar Plot Color', '#1f77b4')
    
        # Ensure the merged_labels column is not categorical
            if dataframe[merged_labels_col].dtype == 'category' or not pd.api.types.is_numeric_dtype(dataframe[merged_labels_col]):
                dataframe[merged_labels_col] = dataframe[merged_labels_col].astype(str)

        # Convert feature columns to numeric if necessary
            for feature in features:
                if not pd.api.types.is_numeric_dtype(dataframe[feature]):
                    dataframe[feature] = pd.to_numeric(dataframe[feature], errors='coerce')

        # Group the data by merged_labels and sum up counts for each feature
            grouped_data = dataframe.groupby(merged_labels_col)[features].sum(numeric_only=True).reset_index()

        # Calculate layout dynamically based on the number of selected features
            n_features = len(features)
            n_cols = st.slider('Select Number of Columns in Layout', min_value=1, max_value=4, value=4)
            n_rows = (n_features // n_cols) + (n_features % n_cols > 0)

        # Create subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
            axes = axes.flatten()  # Flatten the axes array for easier iteration

        # Generate bar plots for selected features
            for i, feature in enumerate(features):
                sns.barplot(x=merged_labels_col, y=feature, data=grouped_data, color=color_choice, ax=axes[i])

            # Customize each subplot
                axes[i].set_title(f'{feature} by {merged_labels_col}')
                axes[i].set_xlabel(merged_labels_col)
                axes[i].set_ylabel('Counts')
                axes[i].tick_params(axis='x', rotation=90)

        # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

        # Adjust layout and show the plot
            plt.tight_layout()
            st.pyplot(fig)


        interactive_barplots(df)
    
    
    show_dash1()


   